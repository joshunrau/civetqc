from civetqc.app import App

from unittest.mock import patch

@patch('sys.argv', ['civetqc', 'this_file_does_not_exist.csv'])
def test():
  App.main()

test()
