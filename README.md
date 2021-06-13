# AD-training

Python codes of my master thesis "Homomorphic Encryption for Augmented Democracy"



## Supervisors

Olivier Pereira

Pierrick Méaux


## Usage
High-level usage example. To be improved.

```python
from Online.Parties import System, MLE, User

#Create System, MLE and User
system = System(problem_parameters)
mlengine = MLE(params, problem_parameters, keys)
alice = User(params, problem_parameters, keys, latent_vector, preference)

#Create first weight v^0
v = system.newQuestion(dimension)

#Online Protocol B
while(not stop):
  grad = alice.GD(v)
  v = mlengine.GD(grad)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
