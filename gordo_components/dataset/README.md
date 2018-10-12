## Public component: `gordo_components.dataset.Dataset`

----

This component should take a config of `kwargs` which will configure the `Dataset` to properly return `X` and `y` from `.get_train()` and an `X` from `.get_test()`


### Example theoretical use:

```python
from gordo_components.dataset import Dataset

config = {
    'machine_id': 'm123',
    'tags'      : ['tag1', 'tag2', 'tag3'],
    'time'      : '10.01.2016'
}

dataset = Dataset(**config)
X, y    = dataset.get_train()
X       = dataset.get_test()
```
