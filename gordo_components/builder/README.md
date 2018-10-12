## Public components: `build_model(output_dir, model_config, data_config)`

----
 `build_model(...)` will take:
 - `output_dir` - `str` where to save the model, default to `/data`
 - `model_config` - `dict` of `kwargs` to initialize the model, requiring `type` key, and any other `kwargs` to be passed to the intended model.
 - `data_config` - `dict` of `kwargs` to initialize `gordo_components.dataset.Dataset` in order to fetch desired data.


This method will take all the above and then build, train and serialize a model to file output.


### Example theoretical use:

```python
from gordo_components.builder import build_model
data_config = {
    'machine_id': 'm123',
    'tags'      : ['tag1', 'tag2', 'tag3'],
    'time'      : '10.01.2016'
}
model_config = {
    'type': 'keras',
}
output_dir = '/data'

buid_model(output_dir, model_config, data_config)
```