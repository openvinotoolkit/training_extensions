import os

os.environ["TT_SELENIUM_HEADLESS"] = "False"

os.environ["TT_API_OTHER_TESTS"] = "True"
os.environ["TT_PERFORMANCE_TESTS"] = "True"
os.environ["TT_UNIT_TESTS"] = "True"
os.environ["TT_COMPONENT_TESTS"] = "False"
os.environ["USE_GPU_FOR_TESTS"] = "True"


os.environ["TT_LOGGING_LEVEL"] = "DEBUG"
os.environ["TT_DATABASE_URL"] = "mongodb://validationreports.sclab.intel.com/impt_component_test_results_components"

os.environ["TT_ENV_USER"] = "devuser"
os.environ["TT_ENV_PASS"] = "P@55wordless"

os.environ["DISPLAY"] = ":10.0"
