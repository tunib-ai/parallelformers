.PHONY: style quality

check_dirs := parallelformers/ tests/

style:
	yapf $(check_dirs) --style "{based_on_style: google, indent_width: 4}" --recursive -i
	isort $(check_dirs)
	flake8 $(check_dirs)

quality:
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
