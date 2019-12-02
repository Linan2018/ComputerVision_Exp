from unittest.mock import patch, mock_open
with patch("builtins.open", mock_open(read_data="data")) as mock_file:
    assert open("path/to/open").read() == "data"
    mock_file.assert_called_with("path/to/open")