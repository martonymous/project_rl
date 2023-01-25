import os, uuid, pickle, glob


def make_directory_for_run(unique_run_id):
    """ Make a directory for this training run. """
    print(f'Preparing training run {unique_run_id}')
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
    os.mkdir(f'./runs/{unique_run_id}')

    
def save_model(model, phase, unique_run_id):
    """ Save models at specific point in time. """
    with open(f'./runs/{unique_run_id}/agent_phase-{phase}.pickle', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    list_of_files = glob.glob('./runs/*') # * means all if need specific format then *.csv
    latest_folder = max(list_of_files, key=os.path.getctime)

    list_of_files = glob.glob(f'{latest_folder}/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)