# Using Google Colab
:label:`sec_colab`

We introduced how to run this book on AWS in :numref:`sec_sagemaker` and :numref:`sec_aws`. Another option is running this book on [Google Colab](https://colab.research.google.com/)
if you have a Google account.

To run the code of a section on Colab, simply click the `Colab` button as shown in :numref:`fig_colab`. 

![Run the code of a section on Colab.](../img/colab.png)
:width:`300px`
:label:`fig_colab`


If it is your first time to run a code cell,
you will receive a warning message as shown in :numref:`fig_colab2`.
Just click "RUN ANYWAY" to ignore it.

![Ignore the warning message by clicking "RUN ANYWAY".](../img/colab-2.png)
:width:`300px`
:label:`fig_colab2`

Next, Colab will connect you to an instance to run the code of this section.
Specifically, if a GPU is needed, 
Colab will be automatically requested 
for connecting to a GPU instance.

If you plan to work through many notebooks, it is often easier to
`git clone` the entire book repository into your Colab session, so that
all chapters, images, and auxiliary files are available on the local
filesystem. This also sidesteps "No such file" errors that can occur
when a notebook tries to load images via relative paths that are not
present when only a single notebook is uploaded.


## Summary

* You can use Google Colab to run each section's code in this book.
* Colab will be requested to connect to a GPU instance if a GPU is needed in any section of this book.


## Exercises

1. Open any section of this book using Google Colab.
1. Edit and run any section that requires a GPU using Google Colab.


[Discussions](https://discuss.d2l.ai/t/424)
