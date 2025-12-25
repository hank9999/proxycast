//! Usage Service - Kiro 用量查询服务
//!
//! 通过调用 AWS Q 的 getUsageLimits API 获取用户的用量信息。
//! 参考 Kir-Manager 项目的 usage/usage.go 实现。

use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use serde::{Deserialize, Serialize};
use std::error::Error;
use uuid::Uuid;

// ============================================================================
// 常量定义
// ============================================================================

/// API 端点
pub const USAGE_LIMITS_URL: &str = "https://q.us-east-1.amazonaws.com/getUsageLimits";

/// Query 参数
pub const ORIGIN_PARAM: &str = "AI_EDITOR";
pub const RESOURCE_TYPE_PARAM: &str = "AGENTIC_REQUEST";

/// HTTP 请求超时（秒）
pub const HTTP_TIMEOUT_SECS: u64 = 10;

// ============================================================================
// API Response 数据模型
// ============================================================================

/// API 响应结构
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageLimitsResponse {
    pub subscription_info: SubscriptionInfo,
    pub usage_breakdown_list: Vec<UsageBreakdown>,
}

/// 订阅信息结构
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscriptionInfo {
    pub subscription_title: String,
    #[serde(rename = "type")]
    pub subscription_type: String,
}

/// 用量明细结构
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageBreakdown {
    pub usage_limit_with_precision: f64,
    pub current_usage_with_precision: f64,
    pub display_name: String,
    #[serde(default)]
    pub free_trial_info: Option<FreeTrialInfo>,
    #[serde(default)]
    pub bonuses: Option<Vec<Bonus>>,
}

/// 免费试用信息
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FreeTrialInfo {
    pub usage_limit_with_precision: f64,
    pub current_usage_with_precision: f64,
    pub free_trial_status: String,
}

/// 奖励额度
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bonus {
    pub bonus_code: String,
    pub usage_limit: f64,
    pub current_usage: f64,
    pub status: String,
}

// ============================================================================
// 计算结果数据模型
// ============================================================================

/// 计算后的用量信息
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageInfo {
    /// 订阅类型名称
    pub subscription_title: String,
    /// 总额度
    pub usage_limit: f64,
    /// 已使用
    pub current_usage: f64,
    /// 余额 = usage_limit - current_usage
    pub balance: f64,
    /// 余额低于 20%
    pub is_low_balance: bool,
}

impl UsageInfo {
    /// 创建空的 UsageInfo
    pub fn empty() -> Self {
        Self::default()
    }
}

// ============================================================================
// URL 构造函数
// ============================================================================

/// 构造 API 请求 URL
///
/// **Property 4: Social Auth URL Construction**
/// **Property 5: IdC Auth URL Construction**
/// **Validates: Requirements 2.1, 2.2**
///
/// - Social 认证: 包含 profileArn 参数
/// - IdC 认证: 不包含 profileArn 参数
pub fn build_usage_api_url(
    auth_method: &str,
    profile_arn: Option<&str>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut url = url::Url::parse(USAGE_LIMITS_URL)?;

    {
        let mut query = url.query_pairs_mut();
        query.append_pair("origin", ORIGIN_PARAM);
        query.append_pair("resourceType", RESOURCE_TYPE_PARAM);

        // Property 4: Social Auth URL Construction
        // 只有 social 类型才加入 profileArn
        if auth_method == "social" {
            match profile_arn {
                Some(arn) if !arn.is_empty() => {
                    query.append_pair("profileArn", arn);
                }
                _ => {
                    return Err("social auth requires profileArn".into());
                }
            }
        }
        // Property 5: IdC Auth URL Construction
        // IdC 类型不包含 profileArn
    }

    Ok(url.to_string())
}

// ============================================================================
// 请求头构造函数
// ============================================================================

/// 构造 API 请求头
///
/// **Property 6: User-Agent Header Format**
/// **Validates: Requirements 4.1, 4.2, 4.3**
///
/// Headers:
/// - User-Agent: aws-sdk-js/1.0.0 ua/2.1 os/{os} lang/js md/nodejs#22.21.1 api/codewhispererruntime#1.0.0 m/N,E KiroIDE-{version}-{machineId}
/// - x-amz-user-agent: aws-sdk-js/1.0.0 KiroIDE-{version}-{machineId}
/// - amz-sdk-invocation-id: UUID
/// - amz-sdk-request: attempt=1; max=1
pub fn build_request_headers(
    access_token: &str,
    kiro_version: &str,
    machine_id: &str,
) -> Result<HeaderMap, Box<dyn Error + Send + Sync>> {
    let mut headers = HeaderMap::new();

    // Authorization header
    let auth_value = format!("Bearer {}", access_token);
    headers.insert("Authorization", HeaderValue::from_str(&auth_value)?);

    // User-Agent header
    // 格式: aws-sdk-js/1.0.0 ua/2.1 os/{os} lang/js md/nodejs#22.21.1 api/codewhispererruntime#1.0.0 m/N,E KiroIDE-{version}-{machineId}
    let os_name = std::env::consts::OS;
    let user_agent = format!(
        "aws-sdk-js/1.0.0 ua/2.1 os/{} lang/js md/nodejs#22.21.1 api/codewhispererruntime#1.0.0 m/N,E KiroIDE-{}-{}",
        os_name, kiro_version, machine_id
    );
    headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);

    // x-amz-user-agent header
    let x_amz_user_agent = format!("aws-sdk-js/1.0.0 KiroIDE-{}-{}", kiro_version, machine_id);
    headers.insert(
        "x-amz-user-agent",
        HeaderValue::from_str(&x_amz_user_agent)?,
    );

    // amz-sdk-invocation-id: 每次请求随机生成 UUID
    headers.insert(
        "amz-sdk-invocation-id",
        HeaderValue::from_str(&Uuid::new_v4().to_string())?,
    );

    // amz-sdk-request header
    headers.insert(
        "amz-sdk-request",
        HeaderValue::from_static("attempt=1; max=1"),
    );

    // Connection header
    headers.insert("Connection", HeaderValue::from_static("close"));

    Ok(headers)
}

/// 构造 User-Agent 字符串（用于测试）
#[cfg(test)]
pub fn build_user_agent(kiro_version: &str, machine_id: &str) -> String {
    let os_name = std::env::consts::OS;
    format!(
        "aws-sdk-js/1.0.0 ua/2.1 os/{} lang/js md/nodejs#22.21.1 api/codewhispererruntime#1.0.0 m/N,E KiroIDE-{}-{}",
        os_name, kiro_version, machine_id
    )
}

/// 构造 x-amz-user-agent 字符串（用于测试）
#[cfg(test)]
pub fn build_x_amz_user_agent(kiro_version: &str, machine_id: &str) -> String {
    format!("aws-sdk-js/1.0.0 KiroIDE-{}-{}", kiro_version, machine_id)
}

// ============================================================================
// API 调用函数
// ============================================================================

/// 调用 AWS Q getUsageLimits API 获取用量信息
///
/// **Validates: Requirements 1.1, 4.4**
///
/// # Arguments
/// * `access_token` - Bearer token
/// * `auth_method` - 认证方式 ("social" 或 "idc")
/// * `profile_arn` - Social 认证需要的 profileArn
/// * `machine_id` - 设备 ID (SHA256 哈希)
/// * `kiro_version` - Kiro 版本号
///
/// # Returns
/// * `Ok(UsageInfo)` - 成功时返回计算后的用量信息
/// * `Err` - 失败时返回错误
pub async fn get_usage_limits(
    access_token: &str,
    auth_method: &str,
    profile_arn: Option<&str>,
    machine_id: &str,
    kiro_version: &str,
) -> Result<UsageInfo, Box<dyn Error + Send + Sync>> {
    // 验证参数
    if access_token.is_empty() {
        return Err("invalid token: missing accessToken".into());
    }

    if machine_id.is_empty() {
        return Err("invalid machineID: empty".into());
    }

    // 构造 URL
    let url = build_usage_api_url(auth_method, profile_arn)?;

    // 构造请求头
    let headers = build_request_headers(access_token, kiro_version, machine_id)?;

    // 创建 HTTP 客户端（带超时）
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(HTTP_TIMEOUT_SECS))
        .build()?;

    // 发送请求
    let response = client.get(&url).headers(headers).send().await?;

    // 检查状态码
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API request failed with status {}: {}", status, body).into());
    }

    // 解析响应
    let usage_response: UsageLimitsResponse = response.json().await?;

    // 计算余额并返回
    Ok(calculate_balance(&usage_response))
}

/// 安全地调用 API 获取用量信息
///
/// **Property 3: Error Handling Graceful Degradation**
/// **Validates: Requirements 1.4**
///
/// 当发生任何错误时，返回空的 UsageInfo 而非 panic
pub async fn get_usage_limits_safe(
    access_token: &str,
    auth_method: &str,
    profile_arn: Option<&str>,
    machine_id: &str,
    kiro_version: &str,
) -> UsageInfo {
    match get_usage_limits(
        access_token,
        auth_method,
        profile_arn,
        machine_id,
        kiro_version,
    )
    .await
    {
        Ok(info) => info,
        Err(e) => {
            tracing::warn!("Failed to get usage limits: {}", e);
            UsageInfo::empty()
        }
    }
}

// ============================================================================
// 余额计算函数
// ============================================================================

/// 低余额阈值 (20%)
pub const LOW_BALANCE_THRESHOLD: f64 = 0.2;

/// 从 API 响应计算余额
///
/// **Property 1: Balance Calculation Correctness**
/// **Validates: Requirements 1.2**
///
/// 计算逻辑：
/// - 总额度 = Σ(usage_limit_with_precision + free_trial_info?.usage_limit_with_precision + Σ(bonuses[].usage_limit))
/// - 总使用 = Σ(current_usage_with_precision + free_trial_info?.current_usage_with_precision + Σ(bonuses[].current_usage))
/// - 余额 = 总额度 - 总使用
pub fn calculate_balance(response: &UsageLimitsResponse) -> UsageInfo {
    calculate_balance_with_threshold(response, LOW_BALANCE_THRESHOLD)
}

/// 从 API 响应计算余额（使用指定阈值）
///
/// threshold: 低余额阈值（0.0 ~ 1.0），例如 0.2 表示余额低于 20% 时为低余额
pub fn calculate_balance_with_threshold(
    response: &UsageLimitsResponse,
    threshold: f64,
) -> UsageInfo {
    let mut total_usage_limit = 0.0;
    let mut total_current_usage = 0.0;

    for breakdown in &response.usage_breakdown_list {
        // 基本额度
        total_usage_limit += breakdown.usage_limit_with_precision;
        total_current_usage += breakdown.current_usage_with_precision;

        // 免费试用额度（如果存在）
        if let Some(ref free_trial) = breakdown.free_trial_info {
            total_usage_limit += free_trial.usage_limit_with_precision;
            total_current_usage += free_trial.current_usage_with_precision;
        }

        // 奖励额度（如果存在）
        if let Some(ref bonuses) = breakdown.bonuses {
            for bonus in bonuses {
                total_usage_limit += bonus.usage_limit;
                total_current_usage += bonus.current_usage;
            }
        }
    }

    let balance = total_usage_limit - total_current_usage;

    // Property 2: Low Balance Detection
    // Validates: Requirements 1.3
    // is_low_balance = (balance / total_usage_limit) < threshold
    let is_low_balance = if total_usage_limit > 0.0 {
        (balance / total_usage_limit) < threshold
    } else {
        false
    };

    UsageInfo {
        subscription_title: response.subscription_info.subscription_title.clone(),
        usage_limit: total_usage_limit,
        current_usage: total_current_usage,
        balance,
        is_low_balance,
    }
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use urlencoding;

    // ========================================================================
    // Arbitrary 生成器
    // ========================================================================

    /// 生成有效的 Bonus
    fn arb_bonus() -> impl Strategy<Value = Bonus> {
        (
            "[a-zA-Z0-9]{4,10}", // bonus_code
            0.0..1000.0f64,      // usage_limit
            0.0..1000.0f64,      // current_usage
            prop_oneof!["ACTIVE", "EXPIRED", "PENDING"],
        )
            .prop_map(|(bonus_code, usage_limit, current_usage, status)| Bonus {
                bonus_code,
                usage_limit,
                current_usage,
                status: status.to_string(),
            })
    }

    /// 生成有效的 FreeTrialInfo
    fn arb_free_trial_info() -> impl Strategy<Value = FreeTrialInfo> {
        (
            0.0..1000.0f64, // usage_limit_with_precision
            0.0..1000.0f64, // current_usage_with_precision
            prop_oneof!["ACTIVE", "EXPIRED"],
        )
            .prop_map(|(usage_limit, current_usage, status)| FreeTrialInfo {
                usage_limit_with_precision: usage_limit,
                current_usage_with_precision: current_usage,
                free_trial_status: status.to_string(),
            })
    }

    /// 生成有效的 UsageBreakdown
    fn arb_usage_breakdown() -> impl Strategy<Value = UsageBreakdown> {
        (
            0.0..1000.0f64,                          // usage_limit_with_precision
            0.0..1000.0f64,                          // current_usage_with_precision
            "[a-zA-Z ]{5,20}",                       // display_name
            prop::option::of(arb_free_trial_info()), // free_trial_info
            prop::option::of(prop::collection::vec(arb_bonus(), 0..3)), // bonuses
        )
            .prop_map(
                |(usage_limit, current_usage, display_name, free_trial_info, bonuses)| {
                    UsageBreakdown {
                        usage_limit_with_precision: usage_limit,
                        current_usage_with_precision: current_usage,
                        display_name,
                        free_trial_info,
                        bonuses,
                    }
                },
            )
    }

    /// 生成有效的 SubscriptionInfo
    fn arb_subscription_info() -> impl Strategy<Value = SubscriptionInfo> {
        (
            prop_oneof!["Free Tier", "Pro", "Enterprise"],
            prop_oneof!["FREE", "PAID", "TRIAL"],
        )
            .prop_map(|(title, sub_type)| SubscriptionInfo {
                subscription_title: title.to_string(),
                subscription_type: sub_type.to_string(),
            })
    }

    /// 生成有效的 UsageLimitsResponse
    fn arb_usage_limits_response() -> impl Strategy<Value = UsageLimitsResponse> {
        (
            arb_subscription_info(),
            prop::collection::vec(arb_usage_breakdown(), 1..5),
        )
            .prop_map(
                |(subscription_info, usage_breakdown_list)| UsageLimitsResponse {
                    subscription_info,
                    usage_breakdown_list,
                },
            )
    }

    // ========================================================================
    // Property 1: Balance Calculation Correctness
    // **Feature: kiro-usage-api, Property 1: Balance Calculation Correctness**
    // **Validates: Requirements 1.2**
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 1: 余额计算正确性
        ///
        /// *For any* UsageLimitsResponse with valid usage breakdown data,
        /// the calculated balance SHALL equal (total_usage_limit - total_current_usage),
        /// where totals include base amounts, free trial amounts, and bonus amounts.
        #[test]
        fn prop_balance_calculation_correctness(response in arb_usage_limits_response()) {
            let result = calculate_balance(&response);

            // 手动计算期望值
            let mut expected_limit = 0.0;
            let mut expected_usage = 0.0;

            for breakdown in &response.usage_breakdown_list {
                expected_limit += breakdown.usage_limit_with_precision;
                expected_usage += breakdown.current_usage_with_precision;

                if let Some(ref ft) = breakdown.free_trial_info {
                    expected_limit += ft.usage_limit_with_precision;
                    expected_usage += ft.current_usage_with_precision;
                }

                if let Some(ref bonuses) = breakdown.bonuses {
                    for bonus in bonuses {
                        expected_limit += bonus.usage_limit;
                        expected_usage += bonus.current_usage;
                    }
                }
            }

            let expected_balance = expected_limit - expected_usage;

            // 使用近似比较（浮点数精度问题）
            let epsilon = 1e-10;
            prop_assert!((result.usage_limit - expected_limit).abs() < epsilon,
                "usage_limit mismatch: got {}, expected {}", result.usage_limit, expected_limit);
            prop_assert!((result.current_usage - expected_usage).abs() < epsilon,
                "current_usage mismatch: got {}, expected {}", result.current_usage, expected_usage);
            prop_assert!((result.balance - expected_balance).abs() < epsilon,
                "balance mismatch: got {}, expected {}", result.balance, expected_balance);
        }
    }

    // ========================================================================
    // Property 2: Low Balance Detection
    // **Feature: kiro-usage-api, Property 2: Low Balance Detection**
    // **Validates: Requirements 1.3**
    // ========================================================================

    /// 生成有效的 UsageInfo（直接生成，用于测试低余额检测）
    fn arb_usage_info() -> impl Strategy<Value = (f64, f64)> {
        // 生成 usage_limit 和 balance，确保 balance <= usage_limit
        (0.01..1000.0f64).prop_flat_map(|usage_limit| {
            // balance 可以是 0 到 usage_limit 之间的任意值
            (Just(usage_limit), 0.0..=usage_limit)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 2: 低余额检测
        ///
        /// *For any* UsageInfo where balance/usage_limit < 0.2 (and usage_limit > 0),
        /// is_low_balance SHALL be true; otherwise it SHALL be false.
        #[test]
        fn prop_low_balance_detection((usage_limit, balance) in arb_usage_info()) {
            // 构造一个简单的响应来测试低余额检测
            let current_usage = usage_limit - balance;
            let response = UsageLimitsResponse {
                subscription_info: SubscriptionInfo {
                    subscription_title: "Test".to_string(),
                    subscription_type: "FREE".to_string(),
                },
                usage_breakdown_list: vec![UsageBreakdown {
                    usage_limit_with_precision: usage_limit,
                    current_usage_with_precision: current_usage,
                    display_name: "Test".to_string(),
                    free_trial_info: None,
                    bonuses: None,
                }],
            };

            let result = calculate_balance(&response);

            // 计算期望的 is_low_balance
            let ratio = balance / usage_limit;
            let expected_low_balance = ratio < LOW_BALANCE_THRESHOLD;

            prop_assert_eq!(
                result.is_low_balance,
                expected_low_balance,
                "is_low_balance mismatch: got {}, expected {} (ratio: {}, threshold: {})",
                result.is_low_balance,
                expected_low_balance,
                ratio,
                LOW_BALANCE_THRESHOLD
            );
        }

        /// Property 2 边界情况: 当 usage_limit 为 0 时，is_low_balance 应为 false
        #[test]
        fn prop_low_balance_zero_limit(current_usage in 0.0..100.0f64) {
            let response = UsageLimitsResponse {
                subscription_info: SubscriptionInfo {
                    subscription_title: "Test".to_string(),
                    subscription_type: "FREE".to_string(),
                },
                usage_breakdown_list: vec![UsageBreakdown {
                    usage_limit_with_precision: 0.0,
                    current_usage_with_precision: current_usage,
                    display_name: "Test".to_string(),
                    free_trial_info: None,
                    bonuses: None,
                }],
            };

            let result = calculate_balance(&response);

            // 当 usage_limit 为 0 时，is_low_balance 应为 false（避免除零）
            prop_assert!(!result.is_low_balance,
                "is_low_balance should be false when usage_limit is 0");
        }
    }

    // ========================================================================
    // Property 4: Social Auth URL Construction
    // **Feature: kiro-usage-api, Property 4: Social Auth URL Construction**
    // **Validates: Requirements 2.1**
    // ========================================================================

    /// 生成有效的 profileArn
    fn arb_profile_arn() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9:/-]{10,50}".prop_map(|s| format!("arn:aws:iam::{}", s))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 4: Social Auth URL 构造
        ///
        /// *For any* request with auth_method="social" and a valid profile_arn,
        /// the request URL SHALL contain `profileArn={profile_arn}` as a query parameter.
        #[test]
        fn prop_social_auth_url_contains_profile_arn(profile_arn in arb_profile_arn()) {
            let url = build_usage_api_url("social", Some(&profile_arn)).unwrap();

            // URL 应该包含 profileArn 参数
            prop_assert!(url.contains("profileArn="),
                "Social auth URL should contain profileArn parameter, got: {}", url);

            // URL 应该包含编码后的 profile_arn 值
            let encoded_arn = urlencoding::encode(&profile_arn);
            prop_assert!(url.contains(&encoded_arn.to_string()),
                "Social auth URL should contain encoded profileArn value '{}', got: {}", encoded_arn, url);

            // URL 应该包含基本参数
            prop_assert!(url.contains("origin=AI_EDITOR"),
                "URL should contain origin parameter, got: {}", url);
            prop_assert!(url.contains("resourceType=AGENTIC_REQUEST"),
                "URL should contain resourceType parameter, got: {}", url);
        }

        /// Property 4 边界情况: Social auth 缺少 profileArn 应返回错误
        #[test]
        fn prop_social_auth_requires_profile_arn(_dummy in 0..10i32) {
            // 测试 None
            let result = build_usage_api_url("social", None);
            prop_assert!(result.is_err(), "Social auth without profileArn should fail");

            // 测试空字符串
            let result = build_usage_api_url("social", Some(""));
            prop_assert!(result.is_err(), "Social auth with empty profileArn should fail");
        }
    }

    // ========================================================================
    // Property 5: IdC Auth URL Construction
    // **Feature: kiro-usage-api, Property 5: IdC Auth URL Construction**
    // **Validates: Requirements 2.2**
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 5: IdC Auth URL 构造
        ///
        /// *For any* request with auth_method="idc",
        /// the request URL SHALL NOT contain `profileArn` as a query parameter.
        #[test]
        fn prop_idc_auth_url_no_profile_arn(profile_arn in arb_profile_arn()) {
            // 即使提供了 profile_arn，IdC 认证也不应该包含它
            let url = build_usage_api_url("idc", Some(&profile_arn)).unwrap();

            // URL 不应该包含 profileArn 参数
            prop_assert!(!url.contains("profileArn"),
                "IdC auth URL should NOT contain profileArn parameter, got: {}", url);

            // URL 应该包含基本参数
            prop_assert!(url.contains("origin=AI_EDITOR"),
                "URL should contain origin parameter, got: {}", url);
            prop_assert!(url.contains("resourceType=AGENTIC_REQUEST"),
                "URL should contain resourceType parameter, got: {}", url);
        }

        /// Property 5: IdC auth 不需要 profileArn
        #[test]
        fn prop_idc_auth_works_without_profile_arn(_dummy in 0..10i32) {
            // IdC 认证不需要 profileArn
            let result = build_usage_api_url("idc", None);
            prop_assert!(result.is_ok(), "IdC auth without profileArn should succeed");

            let url = result.unwrap();
            prop_assert!(!url.contains("profileArn"),
                "IdC auth URL should NOT contain profileArn parameter, got: {}", url);
        }
    }

    // ========================================================================
    // Property 3: Error Handling Graceful Degradation
    // **Feature: kiro-usage-api, Property 3: Error Handling Graceful Degradation**
    // **Validates: Requirements 1.4**
    // ========================================================================

    /// 生成各种错误输入场景
    #[derive(Debug, Clone)]
    enum ErrorScenario {
        EmptyToken,
        EmptyMachineId,
        MissingProfileArn,
        EmptyProfileArn,
        InvalidToken,
    }

    /// 生成错误场景的策略
    fn arb_error_scenario() -> impl Strategy<Value = ErrorScenario> {
        prop_oneof![
            Just(ErrorScenario::EmptyToken),
            Just(ErrorScenario::EmptyMachineId),
            Just(ErrorScenario::MissingProfileArn),
            Just(ErrorScenario::EmptyProfileArn),
            Just(ErrorScenario::InvalidToken),
        ]
    }

    /// 生成随机的有效 token（用于非空 token 场景）
    fn arb_valid_token() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9]{20,50}".prop_map(|s| s)
    }

    /// 生成随机的有效 machine_id（用于非空 machine_id 场景）
    fn arb_valid_machine_id() -> impl Strategy<Value = String> {
        "[a-f0-9]{32,64}".prop_map(|s| s)
    }

    /// 生成随机的有效 kiro_version
    fn arb_valid_kiro_version() -> impl Strategy<Value = String> {
        "[0-9]{1,2}\\.[0-9]{1,2}\\.[0-9]{1,3}".prop_map(|s| s)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 3: 错误处理优雅降级
        ///
        /// *For any* error condition (network error, invalid response, missing token),
        /// the safe wrapper function SHALL return an empty UsageInfo with zero values
        /// instead of panicking.
        #[test]
        fn prop_error_handling_graceful_degradation(
            scenario in arb_error_scenario(),
            valid_token in arb_valid_token(),
            valid_machine_id in arb_valid_machine_id(),
            valid_version in arb_valid_kiro_version(),
        ) {
            // 创建 tokio runtime 来运行异步代码
            let rt = tokio::runtime::Runtime::new().unwrap();

            let result = rt.block_on(async {
                match scenario {
                    ErrorScenario::EmptyToken => {
                        // 空 token 应该导致错误
                        get_usage_limits_safe(
                            "",
                            "social",
                            Some("arn:aws:test"),
                            &valid_machine_id,
                            &valid_version
                        ).await
                    }
                    ErrorScenario::EmptyMachineId => {
                        // 空 machine_id 应该导致错误
                        get_usage_limits_safe(
                            &valid_token,
                            "social",
                            Some("arn:aws:test"),
                            "",
                            &valid_version
                        ).await
                    }
                    ErrorScenario::MissingProfileArn => {
                        // social 认证缺少 profileArn 应该导致错误
                        get_usage_limits_safe(
                            &valid_token,
                            "social",
                            None,
                            &valid_machine_id,
                            &valid_version
                        ).await
                    }
                    ErrorScenario::EmptyProfileArn => {
                        // social 认证空 profileArn 应该导致错误
                        get_usage_limits_safe(
                            &valid_token,
                            "social",
                            Some(""),
                            &valid_machine_id,
                            &valid_version
                        ).await
                    }
                    ErrorScenario::InvalidToken => {
                        // 无效 token 会导致网络错误（401/403）
                        get_usage_limits_safe(
                            "invalid_token_that_will_fail",
                            "idc",
                            None,
                            &valid_machine_id,
                            &valid_version
                        ).await
                    }
                }
            });

            // 无论什么错误场景，safe 函数都应该返回空的 UsageInfo
            prop_assert_eq!(result.usage_limit, 0.0,
                "Error scenario {:?} should return zero usage_limit", scenario);
            prop_assert_eq!(result.current_usage, 0.0,
                "Error scenario {:?} should return zero current_usage", scenario);
            prop_assert_eq!(result.balance, 0.0,
                "Error scenario {:?} should return zero balance", scenario);
            prop_assert!(!result.is_low_balance,
                "Error scenario {:?} should return false is_low_balance", scenario);
            prop_assert!(result.subscription_title.is_empty(),
                "Error scenario {:?} should return empty subscription_title", scenario);
        }
    }

    // 保留原有的单元测试作为补充（快速验证）
    #[tokio::test]
    async fn test_error_handling_empty_token() {
        let result =
            get_usage_limits_safe("", "social", Some("arn:aws:test"), "machine123", "1.0.0").await;
        assert_eq!(result.usage_limit, 0.0);
        assert_eq!(result.balance, 0.0);
    }

    #[tokio::test]
    async fn test_error_handling_empty_machine_id() {
        let result =
            get_usage_limits_safe("token123", "social", Some("arn:aws:test"), "", "1.0.0").await;
        assert_eq!(result.usage_limit, 0.0);
        assert_eq!(result.balance, 0.0);
    }

    // ========================================================================
    // Property 6: User-Agent Header Format
    // **Feature: kiro-usage-api, Property 6: User-Agent Header Format**
    // **Validates: Requirements 4.1, 4.2**
    // ========================================================================

    /// 生成有效的 Kiro 版本号
    fn arb_kiro_version() -> impl Strategy<Value = String> {
        "[0-9]{1,2}\\.[0-9]{1,2}\\.[0-9]{1,3}".prop_map(|s| s)
    }

    /// 生成有效的 Machine ID (SHA256 哈希)
    fn arb_machine_id() -> impl Strategy<Value = String> {
        "[a-f0-9]{64}".prop_map(|s| s)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 6: User-Agent 头格式
        ///
        /// *For any* kiro_version and machine_id strings,
        /// the User-Agent header SHALL match the format:
        /// `aws-sdk-js/1.0.0 ua/2.1 os/{os} lang/js md/nodejs#22.21.1 api/codewhispererruntime#1.0.0 m/N,E KiroIDE-{version}-{machineId}`
        #[test]
        fn prop_user_agent_header_format(
            kiro_version in arb_kiro_version(),
            machine_id in arb_machine_id()
        ) {
            let user_agent = build_user_agent(&kiro_version, &machine_id);

            // 验证格式各部分
            prop_assert!(user_agent.starts_with("aws-sdk-js/1.0.0 ua/2.1 os/"),
                "User-Agent should start with 'aws-sdk-js/1.0.0 ua/2.1 os/', got: {}", user_agent);

            prop_assert!(user_agent.contains("lang/js"),
                "User-Agent should contain 'lang/js', got: {}", user_agent);

            prop_assert!(user_agent.contains("api/codewhispererruntime#1.0.0"),
                "User-Agent should contain 'api/codewhispererruntime#1.0.0', got: {}", user_agent);

            prop_assert!(user_agent.contains("m/N,E"),
                "User-Agent should contain 'm/N,E', got: {}", user_agent);

            // 验证包含 KiroIDE-{version}-{machineId}
            let kiro_suffix = format!("KiroIDE-{}-{}", kiro_version, machine_id);
            prop_assert!(user_agent.ends_with(&kiro_suffix),
                "User-Agent should end with '{}', got: {}", kiro_suffix, user_agent);
        }

        /// Property 6: x-amz-user-agent 头格式
        ///
        /// *For any* kiro_version and machine_id strings,
        /// the x-amz-user-agent header SHALL match the format:
        /// `aws-sdk-js/1.0.0 KiroIDE-{version}-{machineId}`
        #[test]
        fn prop_x_amz_user_agent_header_format(
            kiro_version in arb_kiro_version(),
            machine_id in arb_machine_id()
        ) {
            let x_amz_user_agent = build_x_amz_user_agent(&kiro_version, &machine_id);

            // 验证格式
            let expected = format!("aws-sdk-js/1.0.0 KiroIDE-{}-{}", kiro_version, machine_id);
            prop_assert_eq!(x_amz_user_agent, expected,
                "x-amz-user-agent format mismatch");
        }
    }
}
