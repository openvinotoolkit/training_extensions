import os


def get_theme_path():
    theme_path = os.path.abspath(os.path.dirname(__file__))
    return theme_path


def setup(app):
    theme_path = get_theme_path()
    templates_path = os.path.join(theme_path, 'templates')
    app.config.templates_path.append(templates_path)
    static_path = os.path.join(theme_path, 'static')
    app.config.html_static_path.append(static_path)
    app.add_html_theme('spark_sphinx_theme', theme_path)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}