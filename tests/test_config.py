from finch_agent.config import Settings


def test_empty_secrets_are_treated_as_unset():
    settings = Settings(openai_api_key="", api_auth_token="", checkpoint_database_url="")
    assert settings.openai_api_key is None
    assert settings.api_auth_token is None
    assert settings.checkpoint_database_url is None
