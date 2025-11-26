
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from backend import data_tools

# Mock SessionLocal to avoid actual database calls
@pytest.fixture(autouse=True)
def mock_db_session(mocker):
    mock_session = MagicMock()
    mocker.patch('backend.data_tools.SessionLocal', return_value=mock_session)
    return mock_session

# Mock crud functions
@pytest.fixture
def mock_crud(mocker):
    return mocker.patch('backend.data_tools.crud')

# Mock websocket manager
@pytest.fixture
def mock_manager(mocker):
    """Mock the websocket manager, ensuring broadcast is an async mock."""
    manager_mock = MagicMock()
    manager_mock.broadcast = AsyncMock()
    mocker.patch('backend.data_tools.manager', new=manager_mock)
    return manager_mock

# Mock hardware scanner
@pytest.fixture
def mock_scanner(mocker):
    mock = mocker.patch('backend.data_tools.scanner')
    mock.check_feasibility.return_value = {"feasible": True}
    return mock

# Mock Hugging Face components
@pytest.fixture
def mock_hf(mocker):
    """Mock the Hugging Face tokenizer and model to avoid actual model loading."""
    mock_tokenizer_from_pretrained = mocker.patch('backend.data_tools.AutoTokenizer.from_pretrained')
    mock_model_from_pretrained = mocker.patch('backend.data_tools.AutoModelForCausalLM.from_pretrained')

    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.decode.return_value = "Assistant: A response. Score (1-5): 3"
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer_instance

    mock_model_instance = MagicMock()
    mock_model_instance.generate.return_value = [MagicMock()]
    mock_model_from_pretrained.return_value = mock_model_instance

    return mock_tokenizer_from_pretrained, mock_model_from_pretrained

def test_generate_synthetic_data_task_success(mock_db_session, mock_crud, mock_manager, mock_scanner, mock_hf):
    """Test the successful execution of the synthetic data generation task."""
    model_info = {"hyperparameters": {"topic": "AI", "count": 5}}

    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        data_tools.generate_synthetic_data_task(model_id=1, dataset_id=1, model_info=model_info)

    mock_crud.update_model_status.assert_any_call(mock_db_session, 1, "running")
    mock_manager.broadcast.assert_called()
    mock_hf[0].assert_called_with("state-spaces/mamba-130m-hf")
    mock_hf[1].assert_called_with("state-spaces/mamba-130m-hf")
    mock_open.assert_called_with("uploads/synthetic_data_1.jsonl", "w")
    mock_crud.update_model_status.assert_called_with(
        mock_db_session, 1, "completed",
        metrics={"count": 5, "output_file": "synthetic_data_1.jsonl"},
        model_path="uploads/synthetic_data_1.jsonl"
    )

def test_generate_synthetic_data_task_hw_fail(mock_db_session, mock_crud, mock_manager, mock_scanner):
    """Test hardware feasibility check failure."""
    mock_scanner.check_feasibility.return_value = {"feasible": False}
    data_tools.generate_synthetic_data_task(model_id=2, dataset_id=2, model_info={})

    mock_crud.update_model_status.assert_called_with(mock_db_session, 2, "failed")
    mock_manager.broadcast.assert_called()

def test_llm_judge_task_success(mock_db_session, mock_crud, mock_manager, mock_hf, mocker):
    """Test successful LLM judge task with a corrected newline character."""
    mock_dataset = MagicMock()
    mock_dataset.filename = "judgeme.jsonl"
    mock_crud.get_dataset.return_value = mock_dataset

    # Corrected mock data with a real newline character
    mock_file_content = '{"question": "q1", "answer": "a1"}\n{"question": "q2", "answer": "a2"}'
    mocker.patch('builtins.open', mocker.mock_open(read_data=mock_file_content))

    data_tools.llm_judge_task(model_id=3, dataset_id=3, model_info={})

    mock_crud.get_dataset.assert_called_with(mock_db_session, 3)
    mock_crud.update_model_status.assert_any_call(mock_db_session, 3, "running")
    mock_crud.update_model_status.assert_called_with(
        mock_db_session, 3, "completed",
        metrics={"average_score": 3.0, "judged_count": 2}
    )

def test_llm_judge_task_unsupported_file(mock_db_session, mock_crud, mock_manager):
    """Test that judge task fails with non-jsonl files."""
    mock_dataset = MagicMock()
    mock_dataset.filename = "judgeme.csv"
    mock_crud.get_dataset.return_value = mock_dataset

    data_tools.llm_judge_task(model_id=4, dataset_id=4, model_info={})

    mock_crud.update_model_status.assert_called_with(mock_db_session, 4, "failed")

def test_constitutional_ai_task_success(mock_db_session, mock_crud, mock_manager, mock_hf):
    """Test the successful execution of the constitutional AI data generation task."""
    model_info = {"hyperparameters": {"constitution": "Be nice", "count": 2, "topic": "Ethics"}}

    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        data_tools.generate_constitutional_data_task(model_id=5, dataset_id=5, model_info=model_info)

    mock_crud.update_model_status.assert_any_call(mock_db_session, 5, "running")
    mock_open.assert_called_with("uploads/constitutional_data_5.jsonl", "w")
    mock_crud.update_model_status.assert_called_with(
        mock_db_session, 5, "completed",
        metrics={"count": 2, "output_file": "constitutional_data_5.jsonl"},
        model_path="uploads/constitutional_data_5.jsonl"
    )
