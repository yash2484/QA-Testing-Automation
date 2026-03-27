import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from config import llm_config 

# 1. Data architect and generator agent

data_architect = AssistantAgent(
    name="data_architect",
    llm_config=llm_config,
    system_message="""
    You are a Data Architect specialising in QA test data generation.
    When given a schema, seed rows, and validation rules:

    1. Write a Python script (using only stdlib + csv module) that creates
       a file at: workspace/registration_test_data.csv

    2. The CSV must have EXACTLY these columns:
       username, email, age, password, expected_status, failure_reason

    3. Generate 20 rows total:
       - 8 PASS rows (all validation rules met)
       - 5 FAIL rows for invalid age (< 18)
       - 4 FAIL rows for invalid email (missing @ or missing .com)
       - 3 FAIL rows for short password (< 8 chars)

    4. Base your data on the seed rows provided. Vary names and values
       realistically. Do NOT copy seeds verbatim.

    5. When the script is complete, say exactly: CSV_READY

    IMPORTANT: Always wrap code in ```python blocks.
    """
)

# 2. SDET expert agent 
sdet_expert = AssistantAgent(
    name="sdet_expert",
    llm_config=llm_config,
    system_message="""
    You are a Senior SDET (Software Development Engineer in Test).

    TASK: Write a pytest automation script that tests a user registration API.

    SCRIPT REQUIREMENTS:
    - File must be saved as: workspace/test_registration_api.py
    - Use pandas to read: workspace/registration_test_data.csv
    - For each row, call a local stub function `mock_register_api(row)`
      that simulates the API (define it inside the script).
    - The mock_register_api function must implement these rules:
        * Age < 18 → return 'FAIL'
        * '@' not in email or '.com' not in email → return 'FAIL'
        * len(password) < 8 → return 'FAIL'
        * Otherwise → return 'PASS'
    - Each pytest test function reads the CSV and uses
      @pytest.mark.parametrize to run one test per row.
    - Assert: mock_register_api(row) == row['expected_status']
    - Print a clear PASS/FAIL line for each test case.

    SELF-HEALING: If you receive an error traceback, read it carefully,
    fix the specific issue, rewrite the complete script, and say: SCRIPT_FIXED

    When first script is written, say: SCRIPT_READY
    """
)

# 3. Test runner

test_runner = UserProxyAgent(
    name="test_runner",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False,
        "timeout": 90,
    },
    is_termination_msg=lambda msg: any(
        s in msg.get("content", "")
        for s in ["TESTS_PASSED", "TERMINATE"]
    ),
)


# Groupchat and custom speaker selection workflow

# CUSTOM SPEAKER SELECTOR

def custom_speaker_selector(last_speaker, groupchat):
    msgs = groupchat.messages
    last_msg = msgs[-1].get("content", "").lower() if msgs else ""

    if last_speaker.name == "test_runner":
        if any(s in last_msg for s in ["script_ready", "script_fixed"]):
            return test_runner        # Execute the script
        if "csv_ready" in last_msg:
            return sdet_expert        # CSV done → write tests
        return data_architect         # First turn: generate CSV

    if last_speaker.name == "data_architect":
        return test_runner            # Hand file confirmation back

    if last_speaker.name == "sdet_expert":
        return test_runner            # Execute whatever was written

    return None                       # Terminate

# SELF-HEALING OVERRIDE

# Re-route errors from test_runner back to sdet_expert

_original_selector = custom_speaker_selector
def healing_selector(last_speaker, groupchat):
    msgs = groupchat.messages
    last_msg = msgs[-1].get("content", "").lower() if msgs else ""
    if last_speaker.name == "test_runner":
        errors = ["error", "traceback", "exception", "failed"]
        if any(e in last_msg for e in errors):
            print('[SELF-HEAL] Error detected — routing to sdet_expert')
            return sdet_expert
    return _original_selector(last_speaker, groupchat)

# GROUPCHAT + MANAGER

groupchat = GroupChat(
    agents=[test_runner, data_architect, sdet_expert],
    messages=[],
    max_round=15,
    speaker_selection_method=healing_selector,
    allow_repeat_speaker=True,
)
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

# SEED MESSAGE 

SEED_MESSAGE = """
You are starting the QA pipeline for a User Registration API.

=== SCHEMA ===
Columns: username,email,age,password,expected_status,failure_reason

=== VALIDATION RULES ===
- AGE:      Must be >= 18 to PASS
- EMAIL:    Must contain '@' AND '.com' to PASS
- PASSWORD: Must be >= 8 characters to PASS

=== SEED ROWS ===
username,email,age,password,expected_status,failure_reason
alice_dev,alice@example.com,25,SecurePass1!,PASS,All rules met
bob_smith,bob@company.com,32,P@ssword99,PASS,All rules met
minor_user,minor@example.com,16,MyPassword1,FAIL,Age < 18
teen_edge,teen@example.com,17,ValidPass12,FAIL,Age < 18 boundary
bad_email1,notanemail,28,Password1!,FAIL,Missing @ and .com
bad_email2,user@nodotcom,30,Password1!,FAIL,Missing .com
short_pass,user@example.com,22,abc12,FAIL,Password < 8 chars
multi_fail,bademail,15,short,FAIL,Age+email+password all fail

=== YOUR TASK ===
Write a Python script that generates workspace/registration_test_data.csv
with 20 rows (8 PASS + 12 FAIL). When complete: CSV_READY
"""


# ENTRY POINT

if __name__ == "__main__":
    print('=' * 60)
    print('  AUTONOMOUS DDT FRAMEWORK — STARTING')
    print('=' * 60)
    test_runner.initiate_chat(
        manager,
        message=SEED_MESSAGE,
    )
    print('=' * 60)
    print('  PIPELINE COMPLETE')
    print('=' * 60)