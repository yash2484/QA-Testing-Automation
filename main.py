import os
import autogen
from dotenv import load_dotenv
 
load_dotenv()
 
llm_config = {
    "config_list": [{
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }],
    "temperature": 0.1,
    "timeout": 180,
    "cache_seed": None,
}
 
# Pipeline state machine — mutable so the selector closure can update them
pipeline_state   = ["INIT"]
validation_count = [0]

SEED_MESSAGE = "Automate the QA lifecycle for the User Registration API."

# data architect 

# Change lines 31-32 to this:
data_architect = autogen.AssistantAgent(
    name="data_architect",
    llm_config=llm_config,
    system_message="""
    You are a Data Architect specialising in QA test data generation.
    PHASE 1: Write a Python script to generate workspace/registration_test_data.csv
    with 20 rows (8 PASS, 5 AGE_FAIL, 4 EMAIL_FAIL, 3 PASSWORD_FAIL).
    PHASE 2: Use chromadb.PersistentClient(path="chroma_db") to chunk by test_type, 
    embed with all-MiniLM-L6-v2, and store in collection: registration_test_data.
    When both phases complete, print: DATA_INDEXED
    """
)

# test case gen
# Change lines 43-44 to this:
test_case_generator = autogen.AssistantAgent(
    name="test_case_generator",
    llm_config=llm_config,
    system_message="""
    STEP 1: Use chromadb.PersistentClient(path="chroma_db") to query the 
    collection: registration_test_data and retrieve all 4 chunks. 
    Print: RAG_RETRIEVAL_COMPLETE
    STEP 2: Using retrieved chunks, write workspace/test_registration_api.py
    using pytest + pandas. Implement mock_register_api(row) stub.
    Print: SCRIPT_READY
    """
)

#validator 
validator = autogen.AssistantAgent(
    name="validator",
    llm_config=llm_config,
    system_message="""
    You are a QA Architect. Validate the generated pytest script against:
    1. Column coverage (correct column names)
    2. Business rule implementation in mock_register_api()
    3. Parametrize coverage (all 20 rows)
    4. Assertion quality (descriptive error messages)
    5. RAG alignment (8 PASS, 5 AGE, 4 EMAIL, 3 PASSWORD rows)
    Output SCRIPT_VALIDATED or REVISION_NEEDED on its own line.
    """
)


#test runner
test_runner = autogen.AssistantAgent(
    name="test_runner",
    llm_config=llm_config,
    system_message="""
    Run workspace/test_registration_api.py via subprocess.
    Parse pytest -v output, join with registration_test_data.csv,
    write workspace/test_results.csv with columns:
    test_id, username, email, age, password, expected_status,
    actual_status, test_outcome (MATCH/MISMATCH), failure_reason, test_category.
    Print summary. Print: TESTS_COMPLETE
    """
)

# to keep the pipeline autonomous 
pipeline_runner = autogen.UserProxyAgent(
    name="pipeline_runner",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False,
        "timeout": 300,
        "last_n_messages": 3,
    },
    max_consecutive_auto_reply=25,
    is_termination_msg=lambda m: "TESTS_COMPLETE" in m.get("content", ""),
)



# Speaker Selector 

def qa_pipeline_selector(last_speaker, groupchat):
    messages = groupchat.messages
    last_msg = messages[-1].get('content', '').lower().strip() if messages else ''
    agents   = {a.name: a for a in groupchat.agents}
 
    # Update state from signal words
    if 'data_indexed'     in last_msg: pipeline_state[0] = 'DATA_READY'
    elif 'script_ready'   in last_msg or 'script_revised' in last_msg:
                                        pipeline_state[0] = 'SCRIPT_READY'
    elif 'revision_needed' in last_msg: pipeline_state[0] = 'NEEDS_REVISION'
    elif 'script_validated' in last_msg: pipeline_state[0] = 'VALIDATED'
    elif 'tests_complete'  in last_msg: pipeline_state[0] = 'TESTS_DONE'
 
    state = pipeline_state[0]
 
    if last_speaker.name == 'pipeline_runner':
        if   state == 'INIT':           return agents['data_architect']
        elif state == 'DATA_READY':     return agents['test_case_generator']
        elif state == 'SCRIPT_READY':   return agents['validator']
        elif state == 'NEEDS_REVISION': return agents['test_case_generator']
        elif state == 'VALIDATED':      return agents['test_runner']
 
    elif last_speaker.name in ('data_architect', 'test_case_generator', 'test_runner'):
        return agents['pipeline_runner']   # execute the code first
 
    elif last_speaker.name == 'validator':
        validation_count[0] += 1
        if state == 'VALIDATED' or validation_count[0] >= 3:
            validation_count[0] = 0
            return agents['test_runner']   # approved or max iters hit
        return agents['test_case_generator']   # send back for revision
 
    return agents['pipeline_runner']   # fallback

# groupchat

groupchat = autogen.GroupChat(
    agents=[pipeline_runner, data_architect,
            test_case_generator, validator, test_runner],
    messages=[],
    max_round=35,
    speaker_selection_method=qa_pipeline_selector,
    allow_repeat_speaker=True,
)
 
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)
 
if __name__ == '__main__':
    os.makedirs('workspace', exist_ok=True)
    os.makedirs('workspace/chroma_db', exist_ok=True)
    pipeline_runner.initiate_chat(manager, message=SEED_MESSAGE)

