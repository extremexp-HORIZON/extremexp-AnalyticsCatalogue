__docformat__ = 'numpy'
from os import PathLike
from pathlib import Path
from pdoc.doc import Module
from pdoc.render import html_module, configure
from IPython.display import HTML, display, Markdown

# Hack!
Path('module.html.jinja2').write_text("""
{% extends "default/module.html.jinja2" %}
{% block body %}
    {{ self.content() }}
{% endblock body %}
""")

def fmt_help(obj: object, template_directory ='.')->None:
    """Format and display a docstring"""

    configure(search=False, template_directory=template_directory,
              show_source=False, docformat='numpy')
    display(HTML(html_module(Module(obj),[])))

def include_md(fn: PathLike)->None:
    """Include markdown in a notebook"""
    display(Markdown(Path(fn).read_text()))