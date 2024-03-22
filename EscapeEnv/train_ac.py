import os 
from pathlib import Path


from EscapeEnv.utils import ALGOS, load_yaml, save_yaml, create_output_dir, update_hyperparams, create_parser
from EscapeEnv.common.envs import EscapeEnv
from EscapeEnv.common.callbacks import ActorCriticCallback

def train():
    args = create_parser()
    output_dir = create_output_dir(args)
    
    yaml_dir = os.path.join(Path(__file__).parent.parent, 'hyperparams')
    algo_yaml_path = os.path.join(yaml_dir, args['algo']+'.yml')

    # Load hyperparameters from YAML
    hyperparams = load_yaml(algo_yaml_path)
    # Update hyperparameters with command-line arguments
    update_hyperparams(hyperparams, args['hyperparams'])

    # Save the updated hyperparameters to a new YAML file
    new_yaml_file_path = os.path.join(output_dir, 'hyperparameters.yml')
    save_yaml(hyperparams, new_yaml_file_path)
    
    total_timesteps = hyperparams.pop('total_timesteps')
    callback_kwargs = hyperparams.pop('callback_kwargs')
    env_kwargs = hyperparams.pop('env_kwargs')

    for key, value in hyperparams.items():
        print("{}: {}".format(key, value))
    
    env = EscapeEnv(**env_kwargs)
    # env = EscapeEnv(is_legal_action=True)
    callback = ActorCriticCallback(callback_kwargs=callback_kwargs)
    model_class = ALGOS[args['algo']]
    model = model_class(env=env, **hyperparams, save_path=output_dir, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    

if __name__ == "__main__":
    train()