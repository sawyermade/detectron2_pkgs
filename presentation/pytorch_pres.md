# Pytorch

## Tensor class
The tensor class is a numpy replacement which can run on GPUs.

Has almost identical function and very similar syntax but has slight differences.

### Install torch and torchvision:
```bash
python3 -m pip install -U torch torchvision --user
```

### This imports the torch class which contains the tensor class:
```python
import torch
```

### Create an empty tensor, same as numpy:
```python
x = torch.empty(4, 6)
```
This initializes them to whatever is contained in memory. **NOT GUARANTEED TO BE ZERO!!!**

### Create zero tensor, same as numpy:
```python
# Default dtype is float32
x = torch.zeros(4, 6)
```

### Create ones tensor, same as numpy:
```python
x = torch.ones(4, 6)
```

### Can even set dtype like numpy:
```python
# Default for torch.float is float32
x = torch.zeros(4, 6, dtype=torch.float)

# Get dtype
x.dtype

# Float16
x = torch.ones(4, 6, dtype=torch.float16)
x.dtype
```

### Create tensor from list, similar to np.asarray():
```python
x = torch.tensor([1, 2, 3])
```

### new_ method prefix:
The torch.new_ method prefix takes in a size. It keeps all the same properties as the original tensor but with different dimensions.
```python
x = torch.zeros(4, 6, dtype=torch.float16)

y = x.new_ones(7, 7)

y.dtype
```

### _like method postfix:
This keeps the size but lets you change other features like dtype.
```python
x = torch.ones(4, 6)

y = torch.zeros_like(x, dtype=torch.int)

y.size()

y.dtype
```

### torch.size() method, similar to np.shape
```python
x = torch.zeros(4, 6)

# Prints [4, 6]
x.size()

# Prints 4
x.size()[0]
```


