-- Connect to DB and run the following SQL to regenerate N LiteLLM users in bulk.
-- All stress-test users share one budget record (`stress_test_budget`).

-- 1. DELETE TEST USERS
DELETE FROM "LiteLLM_EndUserTable" WHERE "user_id" LIKE 'stress_test_user_%';

-- 2. Ensure shared budget exists (adjust limits/duration as needed)
WITH upsert_budget AS (
    INSERT INTO "LiteLLM_BudgetTable" (
        "budget_id",
        "max_budget",
        "soft_budget",
        "max_parallel_requests",
        "tpm_limit",
        "rpm_limit",
        "model_max_budget",
        "budget_duration",
        "budget_reset_at",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by"
    )
    VALUES (
        'stress_test_budget',
        1000000000000,          -- generous hard budget
        1000000000000,          -- generous soft budget
        1000000,                -- high parallel requests
        1000000,                -- tokens per minute
        100000,                 -- requests per minute
        NULL,                   -- model-specific budget caps
        INTERVAL '30 days',     -- budget window
        NOW(),
        NOW(),
        'regenerate-llm-users.sql',
        NOW(),
        'regenerate-llm-users.sql'
    )
    ON CONFLICT ("budget_id") DO UPDATE SET
        "max_budget" = EXCLUDED."max_budget",
        "soft_budget" = EXCLUDED."soft_budget",
        "max_parallel_requests" = EXCLUDED."max_parallel_requests",
        "tpm_limit" = EXCLUDED."tpm_limit",
        "rpm_limit" = EXCLUDED."rpm_limit",
        "model_max_budget" = EXCLUDED."model_max_budget",
        "budget_duration" = EXCLUDED."budget_duration",
        "budget_reset_at" = EXCLUDED."budget_reset_at",
        "updated_at" = NOW(),
        "updated_by" = EXCLUDED."updated_by"
    RETURNING "budget_id"
),
budget_row AS (
    SELECT "budget_id" FROM upsert_budget
    UNION ALL
    SELECT "budget_id" FROM "LiteLLM_BudgetTable" WHERE "budget_id" = 'stress_test_budget'
    LIMIT 1
)

-- 3. BULK INSERT N TEST USERS USING generate_series()
-- Replace 1_000_000 below with your desired number of users (N).
INSERT INTO "LiteLLM_EndUserTable"
    ("user_id", "alias", "spend", "allowed_model_region", "default_model", "budget_id", "blocked")
SELECT
    'stress_test_user_' || gs.id::text AS user_id,
    NULL,                     -- alias
    0.0,                      -- spend
    NULL,                     -- allowed_model_region
    NULL,                     -- default_model
    b."budget_id",            -- shared budget for all stress users
    FALSE                     -- blocked
FROM generate_series(1, 1000000) AS gs(id)
CROSS JOIN budget_row b;

-- Optional: Verify the count
SELECT COUNT(*) FROM "LiteLLM_EndUserTable" WHERE "user_id" LIKE 'stress_test_user_%';
