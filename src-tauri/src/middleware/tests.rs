//! Middleware 模块属性测试
//!
//! 使用 proptest 进行属性测试

use crate::config::RemoteManagementConfig;
use crate::middleware::management_auth::{
    clear_auth_failure_state, ManagementAuthLayer, ManagementAuthService,
};
use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, Response, StatusCode},
};
use proptest::prelude::*;
use std::net::SocketAddr;
use std::task::{Context, Poll};
use tower::{Layer, Service};

/// 生成随机的 secret_key（非空）
fn arb_secret_key() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_-]{8,32}".prop_map(|s| s)
}

/// 生成随机的无效 secret_key（与有效 key 不同）
fn arb_invalid_secret_key(valid_key: String) -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_-]{8,32}".prop_filter_map("must differ from valid key", move |s| {
        if s != valid_key {
            Some(s)
        } else {
            None
        }
    })
}

/// 生成随机的 IP 地址
fn arb_ip_addr() -> impl Strategy<Value = String> {
    prop_oneof![
        // localhost IPv4
        Just("127.0.0.1".to_string()),
        // localhost IPv6
        Just("::1".to_string()),
        // remote IPv4
        (1u8..255u8, 0u8..255u8, 0u8..255u8, 1u8..255u8).prop_filter_map(
            "not localhost",
            |(a, b, c, d)| {
                if a == 127 {
                    None
                } else {
                    Some(format!("{}.{}.{}.{}", a, b, c, d))
                }
            }
        ),
    ]
}

/// 生成随机端口
fn arb_port() -> impl Strategy<Value = u16> {
    1024u16..65535u16
}

/// Mock service that always returns 200 OK
#[derive(Clone)]
struct MockService;

impl Service<Request<Body>> for MockService {
    type Response = Response<Body>;
    type Error = std::convert::Infallible;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: Request<Body>) -> Self::Future {
        Box::pin(async {
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("success"))
                .unwrap())
        })
    }
}

/// Helper to create a request with optional Authorization header
fn create_request_with_auth(auth_header: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().uri("/v0/management/status");

    if let Some(auth) = auth_header {
        builder = builder.header("authorization", auth);
    }

    builder.body(Body::empty()).unwrap()
}

/// Helper to create a request with X-Management-Key header
fn create_request_with_management_key(key: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().uri("/v0/management/status");

    if let Some(k) = key {
        builder = builder.header("x-management-key", k);
    }

    builder.body(Body::empty()).unwrap()
}

/// Helper to create a request with X-Management-Key and X-Forwarded-For headers
fn create_request_with_management_key_and_forwarded(
    key: Option<&str>,
    forwarded_for: Option<&str>,
) -> Request<Body> {
    let mut builder = Request::builder().uri("/v0/management/status");

    if let Some(k) = key {
        builder = builder.header("x-management-key", k);
    }

    if let Some(addr) = forwarded_for {
        builder = builder.header("x-forwarded-for", addr);
    }

    builder.body(Body::empty()).unwrap()
}

#[test]
fn test_management_auth_rate_limit_after_failures() {
    clear_auth_failure_state();
    let config = RemoteManagementConfig {
        allow_remote: true,
        secret_key: Some("valid_key".to_string()),
        disable_control_panel: false,
    };
    let layer = ManagementAuthLayer::new(config);
    let mut service = layer.layer(MockService);
    let rt = tokio::runtime::Runtime::new().unwrap();

    // 使用唯一的 IP 地址避免测试间干扰
    // 使用时间戳和进程ID组合来确保唯一性
    let unique_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let client_ip = format!(
        "203.0.113.{}",
        (unique_id ^ std::process::id() as u64) % 256
    );
    let addr: SocketAddr = format!("{}:12345", client_ip).parse().unwrap();

    for _ in 0..5 {
        let mut req = create_request_with_management_key(Some("invalid"));
        // 安全修复后不再信任 X-Forwarded-For，需要注入 ConnectInfo
        req.extensions_mut().insert(ConnectInfo(addr));
        let response = rt.block_on(async { service.call(req).await.unwrap() });
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    let mut req = create_request_with_management_key(Some("invalid"));
    req.extensions_mut().insert(ConnectInfo(addr));
    let response = rt.block_on(async { service.call(req).await.unwrap() });
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: cliproxyapi-parity, Property 19: Management Auth Rejection**
    /// *For any* management API request without valid secret_key, the response SHALL be 401 Unauthorized.
    /// **Validates: Requirements 9.3**
    #[test]
    fn prop_management_auth_rejection_missing_key(
        secret_key in arb_secret_key()
    ) {
        clear_auth_failure_state();
        // Create config with a valid secret_key
        let config = RemoteManagementConfig {
            allow_remote: true,
            secret_key: Some(secret_key),
            disable_control_panel: false,
        };

        // Create the auth layer and service
        let layer = ManagementAuthLayer::new(config);
        let mut service = layer.layer(MockService);

        // Create request WITHOUT any auth header
        let req = create_request_with_auth(None);

        // Execute the service
        let rt = tokio::runtime::Runtime::new().unwrap();
        let response = rt.block_on(async {
            service.call(req).await.unwrap()
        });

        // Verify: should return 401 Unauthorized
        prop_assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "Request without secret_key should return 401 Unauthorized"
        );
    }

    /// **Feature: cliproxyapi-parity, Property 19: Management Auth Rejection**
    /// *For any* management API request with invalid secret_key, the response SHALL be 401 Unauthorized.
    /// **Validates: Requirements 9.3**
    #[test]
    fn prop_management_auth_rejection_invalid_key(
        secret_key in arb_secret_key()
    ) {
        clear_auth_failure_state();
        // Create config with a valid secret_key
        let config = RemoteManagementConfig {
            allow_remote: true,
            secret_key: Some(secret_key.clone()),
            disable_control_panel: false,
        };

        // Create the auth layer and service
        let layer = ManagementAuthLayer::new(config);
        let mut service = layer.layer(MockService);

        // Create request with WRONG auth header (append "wrong" to make it different)
        let wrong_key = format!("{}wrong", secret_key);
        let req = create_request_with_auth(Some(&format!("Bearer {}", wrong_key)));

        // Execute the service
        let rt = tokio::runtime::Runtime::new().unwrap();
        let response = rt.block_on(async {
            service.call(req).await.unwrap()
        });

        // Verify: should return 401 Unauthorized
        prop_assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "Request with invalid secret_key should return 401 Unauthorized"
        );
    }

    /// **Feature: cliproxyapi-parity, Property 19: Management Auth Rejection**
    /// *For any* management API request with valid secret_key, the response SHALL NOT be 401 Unauthorized.
    /// **Validates: Requirements 9.3**
    #[test]
    fn prop_management_auth_acceptance_valid_key(
        secret_key in arb_secret_key()
    ) {
        clear_auth_failure_state();
        // Create config with a valid secret_key
        let config = RemoteManagementConfig {
            allow_remote: true,
            secret_key: Some(secret_key.clone()),
            disable_control_panel: false,
        };

        // Create the auth layer and service
        let layer = ManagementAuthLayer::new(config);
        let mut service = layer.layer(MockService);

        // Create request with CORRECT auth header
        let req = create_request_with_auth(Some(&format!("Bearer {}", secret_key)));

        // Execute the service
        let rt = tokio::runtime::Runtime::new().unwrap();
        let response = rt.block_on(async {
            service.call(req).await.unwrap()
        });

        // Verify: should return 200 OK (passed through to MockService)
        prop_assert_eq!(
            response.status(),
            StatusCode::OK,
            "Request with valid secret_key should pass through (200 OK)"
        );
    }

    /// **Feature: cliproxyapi-parity, Property 19: Management Auth Rejection**
    /// *For any* management API request with valid X-Management-Key header, the response SHALL NOT be 401 Unauthorized.
    /// **Validates: Requirements 9.3**
    #[test]
    fn prop_management_auth_acceptance_x_management_key(
        secret_key in arb_secret_key()
    ) {
        clear_auth_failure_state();
        // Create config with a valid secret_key
        let config = RemoteManagementConfig {
            allow_remote: true,
            secret_key: Some(secret_key.clone()),
            disable_control_panel: false,
        };

        // Create the auth layer and service
        let layer = ManagementAuthLayer::new(config);
        let mut service = layer.layer(MockService);

        // Create request with X-Management-Key header
        let req = create_request_with_management_key(Some(&secret_key));

        // Execute the service
        let rt = tokio::runtime::Runtime::new().unwrap();
        let response = rt.block_on(async {
            service.call(req).await.unwrap()
        });

        // Verify: should return 200 OK (passed through to MockService)
        prop_assert_eq!(
            response.status(),
            StatusCode::OK,
            "Request with valid X-Management-Key should pass through (200 OK)"
        );
    }

    /// **Feature: cliproxyapi-parity, Property 19: Management Auth Rejection**
    /// *For any* management API request with invalid X-Management-Key header, the response SHALL be 401 Unauthorized.
    /// **Validates: Requirements 9.3**
    #[test]
    fn prop_management_auth_rejection_invalid_x_management_key(
        secret_key in arb_secret_key()
    ) {
        clear_auth_failure_state();
        // Create config with a valid secret_key
        let config = RemoteManagementConfig {
            allow_remote: true,
            secret_key: Some(secret_key.clone()),
            disable_control_panel: false,
        };

        // Create the auth layer and service
        let layer = ManagementAuthLayer::new(config);
        let mut service = layer.layer(MockService);

        // Create request with WRONG X-Management-Key header
        let wrong_key = format!("{}wrong", secret_key);
        let req = create_request_with_management_key(Some(&wrong_key));

        // Execute the service
        let rt = tokio::runtime::Runtime::new().unwrap();
        let response = rt.block_on(async {
            service.call(req).await.unwrap()
        });

        // Verify: should return 401 Unauthorized
        prop_assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "Request with invalid X-Management-Key should return 401 Unauthorized"
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[tokio::test]
    async fn test_auth_rejection_no_header() {
        let config = RemoteManagementConfig {
            allow_remote: true,
            secret_key: Some("test-secret-key".to_string()),
            disable_control_panel: false,
        };

        let layer = ManagementAuthLayer::new(config);
        let mut service = layer.layer(MockService);

        let req = create_request_with_auth(None);
        let response = service.call(req).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_rejection_wrong_key() {
        let config = RemoteManagementConfig {
            allow_remote: true,
            secret_key: Some("correct-key".to_string()),
            disable_control_panel: false,
        };

        let layer = ManagementAuthLayer::new(config);
        let mut service = layer.layer(MockService);

        let req = create_request_with_auth(Some("Bearer wrong-key"));
        let response = service.call(req).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_acceptance_correct_key() {
        let config = RemoteManagementConfig {
            allow_remote: true,
            secret_key: Some("correct-key".to_string()),
            disable_control_panel: false,
        };

        let layer = ManagementAuthLayer::new(config);
        let mut service = layer.layer(MockService);

        let req = create_request_with_auth(Some("Bearer correct-key"));
        let response = service.call(req).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
