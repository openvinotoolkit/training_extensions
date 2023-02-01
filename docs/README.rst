#######################
Contribution Guidelines
#######################

<README to be updated>

Documentation is written in reStructuredText (.rst), a lightweight markup language, similar to Markdown. However, there is a significant difference between the two in terms of syntax and semantics. In this README, we will skip explaining the difference between .rst and .md. All you need to know (for now) is that a documentation generator of our choice, Sphinx, natively supports .rst. Hence, all .rst files are a default format by design.


.. note::

	As per our CEO's, Pat Gelsinger, request to unify developer experience, Intel executives are driving a change in the way documentation is generated and published. Sphinx, a documentation generator, is widely used at Intel and a set of custom tools and extensions for Sphinx have been developed to enable documentation on Intel Developer Zone. There is still a host of unanswered questions but what we do know is that Sphinx is not going away and Intel will keep relying on Sphinx.

************************
Adding or editing a file
************************

To add a new file or a heading to an existing .rst file, you need to adhere to a convention created by the author throughout entire documentation. The convention may be customarily created but you must be consistent and use the same convention in every file for Sphinx to read and apply the rules you created. Nevertheless, this documentation follows the convention below.

#. ``#`` with overline
#. ``*`` with overline
#. ``=``
#. ``-``
#. ``^``
#. ``"``

.. code-block::

	##################
	H1: document title
	##################

	*********
	Sample H2
	*********

	Sample H3
	=========

	Sample H4
	---------

	Sample H5
	^^^^^^^^^

	Sample H6
	"""""""""


When adding a new text file (.rst), remember to index the new file in one of the main folders and their index page:

* ``get_started`` >> ``installation.rst``
* ``get_started`` >> ``introduction.rst``
* ``get_started`` >> ``quick_start``

Some folders have a tree structure, which means you need to index the new pages in one of the subfolders, e.g.

* ``tutorials`` >> ``beginner`` >> ``create_dataset.rst``

Once you find the right index file, add the name of the rst file into the toctree directive.

An example of the index page for create_dataset.rst

.. code-block::

	########
	Beginner
	########

	.. toctree::
	  :maxdepth: 3
	  :caption: Contents:

	  <your rst file>

Make sure you follow the indentation pattern. Sphinx is indentation-sensitive.

**************************
Test documentation locally
**************************

To test documentation locally, clone this repository or download the ``docs`` directory and run the following commands:

.. code-block::

	pip install -U sphinx
	pip install pydata-sphinx-theme
	pip install sphinx-panels
	pip install sphinx-copybutton
	pip install sphinx-tabs

In the ``conf.py``, we have already added the variables containing the *pydata_sphinx_theme*, Intel's brand styling, and a couple of extensions.

.. code-block::

	html_theme = "pydata_sphinx_theme"
	html_logo = "_static/logos/otx-logo-black-mini.png"
	html_css_files = [
	    '_static/css/custom.css',
	]

    extensions = [
        'sphinx_panels',
        'sphinx_copybutton',
        'sphinx_tabs.tabs'
    ]

Change the directory to ``docs`` and run ``make html`` to generate documentation. For more information on how to install and use Sphinx, read official `Sphinx documentation <https://www.sphinx-doc.org/en/master/index.html>`_.

***************
RST cheat sheet
***************

The first rule of being efficient and productive is to reuse what has already been created and avoid reinventing the wheel. That's why, you will not find a long shopping list of RST syntax examples here. You can already find them on the internet. Although, we will provide you with a handful of links to explore RST.

* `Cheat sheet no. 1 <https://bashtage.github.io/sphinx-material/rst-cheatsheet/rst-cheatsheet.html#>`_
* `Cheat sheet no. 2 <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_
* `Cheat sheet no. 3 <https://docutils.sourceforge.io/docs/ref/rst/directives.html>`_
* `Cheat sheet no. 4 <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_

******************
Writing guidelines
******************

Authoring a technical piece of writing follows some general rules. Writing clear, concise, and factual sentences in active voice should constitute the vast majority of all sentences in a technical text.

To understand the gist of good technical writing, consider the difference in these two texts:

.. code-block::

	Helen, thy beauty is to me
   Like those Nicean barks of yore,
   That gently, over a perfumed sea,
   The weary, way-worn wanderer bore
   To his own native shore.
		                   Edgar Allan Poe

In technical writing, the text above can be summarized in one sentence.

.. code-block::

  He thinks Helen is beautiful.

As a primer into technical writing, read Google's general guidelines on technical writing:

* `Technical Writing One <https://developers.google.com/tech-writing/one>`_
* `Technical Writing Two <https://developers.google.com/tech-writing/two>`_

The course will give you the essentials for writing good documentation, which is what you need to contribute to this documentation. Your suggestions and commits will undergo a UX review from Adam Czapski or other member of the DX Team to ensure your writing blends into the coherenece and cohesion of existing documentation.

*******************
Contact the DX Team
*******************

If you come across any problems or have any questions regarding documentation, do not hesitate to contact the DX Team. You can reach us by sending an e-mail to `iotg.dx.pl@intel.com`_.

.. _iotg.dx.pl@intel.com: iotg.dx.pl@intel.com
