# AD-training

Python codes of my master thesis "Homomorphic Encryption for Augmented Democracy". https://dial.uclouvain.be/memoire/ucl/en/object/thesis%3A30685


## Supervisors

- Olivier Pereira
- Pierrick Méaux

## Data
Unfortunately Politools did not allowed to share publicly their dataset. Please use with own dataset or contact https://www.smartvote.ch/.

## Libraries used

- sklearn https://github.com/scikit-learn/scikit-learn
- pyfhe https://github.com/sarojaerabelli/py-fhe

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
Pull requests are welcome. 


## License
[MIT](https://choosealicense.com/licenses/mit/)
