Quick start
-----------

The concept of Gordo is to (as of now) process, only, *timeseries*
datasets which are comprised of sensors/tag identifies. The workflow
launches the collection of these tags, building of a defined model and
subsequent deployment of a ML Server which acts as a REST interface
in front of the model.

A typical config file might look like this:

.. literalinclude:: ../../examples/config.yaml