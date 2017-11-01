"""pybnf.algorithms: contains the Algorithm class and subclasses as well as support classes and functions"""


from distributed import as_completed
from distributed import Client
import os
from os import mkdir
from os import chdir
from os import getcwd
from subprocess import run
from subprocess import CalledProcessError
from subprocess import PIPE
from subprocess import STDOUT

from .data import Data
from .pset import PSet
from .pset import Trajectory
from .pset import Model
import numpy as np

import logging
import time


class Result(object):
    """
    Container for the results of a single evaluation in the fitting algorithm
    """

    def __init__(self, paramset, simdata, log):
        """
        Instantiates a Result

        :param paramset: The parameters corresponding to this evaluation
        :type paramset: PSet
        :param simdata: The simulation results corresponding to this evaluation
        :type simdata: list of Data instances
        :param log: The stdout + stderr of the simulations
        :type log: list of str
        """
        self.pset = paramset
        self.simdata = simdata
        self.log = log
        self.score = None  # To be set later when the Result is scored.


class FailedSimulation(object):
    def __init__(self, i):
        self.id = i


class Job:
    """
    Container for information necessary to perform a single evaluation in the fitting algorithm
    """

    def __init__(self, models, params, id, bngcommand, output_dir):
        """
        Instantiates a Job

        :param models: The models to evaluate
        :type models: list of Model instances
        :param params: The parameter set with which to evaluate the model
        :type params: PSet
        :param id: Job identification
        :type id: int
        :param bngcommand: Command to run BioNetGen
        :type bngcommand: str
        :param output_dir path to the directory where I should create my simulation folder
        :type output_dir: str
        """
        self.models = models
        self.params = params
        self.id = id
        self.bng_program = bngcommand
        self.output_dir = output_dir
        self.home_dir = getcwd()

    def _name_with_id(self, model):
        return '%s_%s' % (model.name, self.id)

    def _write_models(self):
        """Writes models to file"""
        model_files = []
        for i, model in enumerate(self.models):
            model_file_name = self._name_with_id(model) + ".bngl"
            model_with_params = model.copy_with_param_set(self.params)
            model_with_params.save(model_file_name)
            model_files.append(model_file_name)
        return model_files

    def run_simulation(self):
        """Runs the simulation and reads in the result"""

        folder = '%s/sim_%s' % (self.output_dir, self.id)
        mkdir(folder)
        try:
            chdir(folder)
            model_files = self._write_models()
            log = self.execute(model_files)
            simdata = self.load_simdata()
            chdir(self.home_dir)
            return Result(self.params, simdata, log)
        except CalledProcessError:
            return FailedSimulation(self.id)

    def execute(self, models):
        """Executes model simulations"""
        log = []
        for model in models:
            cmd = '%s %s' % (self.bng_program, model)
            cp = run(cmd, shell=True, check=True, stderr=STDOUT, stdout=PIPE)
            log.append(cp.stdout)
        return log

    def load_simdata(self):
        """
        Function to load simulation data after executing all simulations for an evaluation

        Returns a nested dictionary structure.  Top-level keys are model names and values are
        dictionaries whose keys are action suffixes and values are Data instances

        :return: dict of dict
        """
        ds = {}
        for model in self.models:
            ds[model.name] = {}
            for suff in model.suffixes:
                if suff[0] == 'simulate':
                    data_file = '%s_%s.gdat' % (self._name_with_id(model), suff[1])
                    data = Data(file_name=data_file)
                else:  # suff[0] == 'parameter_scan'
                    data_file = '%s_%s.scan' % (self._name_with_id(model), suff[1])
                    data = Data(file_name=data_file)
                ds[model.name][suff[1]] = data
        return ds


class Algorithm(object):
    def __init__(self, config):
        """
        Instantiates an Algorithm with a Configuration object.  Also initializes a
        Trajectory instance to track the fitting progress, and performs various additional
        configuration that is consistent for all algorithms

        :param config: The fitting configuration
        :type config: Configuration
        """
        self.config = config
        self.exp_data = self.config.exp_data
        self.objective = self.config.obj
        self.trajectory = Trajectory()
        self.job_id_counter = 0
        self.output_counter = 0

        # Store a list of all Model objects. Change this as needed for compatibility with other parts
        self.model_list = list(self.config.models.values())

        # Generate a list of variable names
        self.variables = self.config.variables

        # Set the space (log or regular) in which each variable moves, as well as the box constraints on the variable.
        # Currently, this is set based on what distribution the variable is initialized with, but these could be made
        # into a separate, custom options
        self.variable_space = dict()  # Contains tuples (space, min_value, max_value)
        for v in self.config.variables_specs:
            if v[1] == 'random_var':
                self.variable_space[v[0]] = ('regular', v[2], v[3])
            elif v[1] == 'lognormrandom_var':
                self.variable_space[v[0]] = ('log', 0., np.inf)  # Questionable if this is the behavior we want.
            elif v[1] == 'loguniform_var':
                self.variable_space[v[0]] = ('log', v[2], v[3])
            elif v[1] == 'static_list_var':
                self.variable_space[v[0]] = ('static', )  # Todo: what is the actual way to mutate this type of param?
            else:
                raise RuntimeError('Unrecognized variable type: %s' % v[1])

    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.

        :return: list of PSets
        """
        logging.info("Initializing algorithm")
        raise NotImplementedError("Subclasses must implement start_run()")

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data

        :param res: result from the completed simulation
        :type res: Result
        :return: List of PSet(s) to be run next.
        """
        logging.info("Retrieved result")
        raise NotImplementedError("Subclasses must implement got_result()")

    def add_to_trajectory(self, res):
        """
        Evaluates the objective function for a Result, and adds the information from the Result to the Trajectory
        instance"""

        score = self.objective.evaluate_multiple(res.simdata, self.exp_data)
        res.score = score
        self.trajectory.add(res.pset, score)

    def random_pset(self):
        """
        Generates a random PSet based on the distributions and bounds for each parameter specified in the configuration

        :return:
        """
        param_dict = dict()
        for (name, type, val1, val2) in self.config.variables_specs:
            if type == 'random_var':
                param_dict[name] = np.random.uniform(val1, val2)
            elif type == 'loguniform_var':
                param_dict[name] = 10.**np.random.uniform(np.log10(val1), np.log10(val2))
            elif type == 'lognormrandom_var':
                param_dict[name] = 10.**np.random.normal(val1, val2)
            elif type == 'static_list_var':
                param_dict[name] = np.random.choice(val1)
            else:
                raise RuntimeError('Unrecognized variable type: %s' % type)
        return PSet(param_dict)

    def add(self, paramset, param, value):
        """
        Helper function to add a value to a param in a parameter set,
        taking into account
        1) Whether this parameter is to be moved in regular or log space
        2) Box constraints on the parameter
        :param paramset:
        :type paramset: PSet
        :param param: name of the parameter
        :type param: str
        :param value: value to be added
        :type value: float
        :return: The result of the addition
        """
        if self.variable_space[param][0] == 'regular':
            return max(self.variable_space[param][1], min(self.variable_space[param][2], paramset[param] + value))
        elif self.variable_space[param][0] == 'log':
            return max(self.variable_space[param][1], min(self.variable_space[param][2],
                                                          10.**(np.log10(paramset[param]) + value)))
        elif self.variable_space[param][0] == 'static':
            return paramset[param]
        else:
            raise RuntimeError('Unrecognized variable space type: %s' % self.variable_space[param][0])

    def diff(self, paramset1, paramset2, param):
        """
        Helper function to calculate paramset1[param] - paramset2[param], taking into account whether
        param is in regular or log space
        """
        if self.variable_space[param][0] == 'regular':
            return paramset1[param] - paramset2[param]
        elif self.variable_space[param][0] == 'log':
            return np.log10(paramset1[param] / paramset2[param])
        elif self.variable_space[param][0] == 'static':
            return 0.  # Don't know what to do here...
        else:
            raise RuntimeError('Unrecognized variable space type: %s' % self.variable_space[param][0])


    def make_job(self, params):
        """
        Creates a new Job using the specified params, and additional specifications that are already saved in the
        Algorithm object

        :param params:
        :type params: PSet
        :return: Job
        """
        self.job_id_counter += 1
        return Job(self.model_list, params, self.job_id_counter, self.config.config['bng_command'],
                   self.config.config['output_dir']+'/Simulations/')

    def output_results(self, name=''):
        """
        Tells the Trajectory to output a log file now with the current best fits.

        This should be called periodically by each Algorithm subclass, and is called by the Algorithm class at the end
        of the simulation.
        :return:
        :param name: Custom string to add to the saved filename. If omitted, we just use a running counter of the
        number of times we've outputted.
        :type name: str
        """
        if name == '':
            name = str(self.output_counter)
        self.output_counter += 1
        filepath = '%s/Results/sorted_params_%s.txt' % (self.config.config['output_dir'], name)
        self.trajectory.write_to_file(filepath)

        # If the user has asked for fewer output files, each time we're here, move the new file to
        # Results/sorted_params.txt, overwriting the previous one.
        if self.config.config['delete_old_files'] == 1:
            noname_filepath = '%s/Results/sorted_params.txt' % self.config.config['output_dir']
            if os.path.isfile(noname_filepath):
                os.remove(noname_filepath)
            os.rename(filepath, noname_filepath)

    def run(self):
        """Main loop for executing the algorithm"""
        client = Client()
        psets = self.start_run()
        jobs = [self.make_job(p) for p in psets]
        futures = [client.submit(job.run_simulation) for job in jobs]
        pending = set(futures)
        pool = as_completed(futures, with_results=True)
        while True:
            f, res = next(pool)
            pending.remove(f)
            self.add_to_trajectory(res)
            response = self.got_result(res)
            if response == 'STOP':
                logging.info("Stop criterion satisfied")
                break
            else:
                new_jobs = [self.make_job(ps) for ps in response]
                new_futures = [client.submit(j.run_simulation) for j in new_jobs]
                pending.update(new_futures)
                pool.update(new_futures)
        client.cancel(list(pending))
        logging.debug("Pending jobs cancelled")
        client.close()
        self.output_results('final')
        logging.info("Fitting complete!")


class ParticleSwarm(Algorithm):
    """
    Implements particle swarm optimization.

    The implementation roughly follows Moraes et al 2015, although is reorganized to better suit PyBNF's format.
    Note the global convergence criterion discussed in that paper is not used (would require too long a
    computation), and instead uses ????

    """

    def __init__(self, config):

        # Former params that are now part of the config
        #variable_list, num_particles, max_evals, cognitive=1.5, social=1.5, w0=1.,
        #wf=0.1, nmax=30, n_stop=np.inf, absolute_tol=0., relative_tol=0.)
        """
        Initial configuration of particle swarm optimizer
        :param conf_dict: The fitting configuration
        :type conf_dict: Configuration

        The config should contain the following definitions:

        population_size - Number of particles in the swarm
        max_iterations - Maximum number of iterations. More precisely, the max number of simulations run is this times
        the population size.
        cognitive - Acceleration toward the particle's own best
        social - Acceleration toward the global best
        particle_weight - Inertia weight of the particle (default 1)

        The following config parameters relate to the complicated method presented is Moraes et al for adjusting the
        inertia weight as you go. These are optional, and this feature will be disabled (by setting
        particle_weight_final = particle_weight) if these are not included.
        It remains to be seen whether this method is at all useful for our applications.

        particle_weight_final -  Inertia weight at the end of the simulation
        adaptive_n_max - Controls how quickly we approach wf - After nmax "unproductive" iterations, we are halfway from
        w0 to wf
        adaptive_n_stop - nd the entire run if we have had this many "unproductive" iterations (should be more than
        adaptive_n_max)
        adaptive_abs_tol - Tolerance for determining if an iteration was "unproductive". A run is unproductive if the
        change in global_best is less than absolute_tol + relative_tol * global_best
        adaptive_rel_tol - Tolerance 2 for determining if an iteration was "unproductive" (see above)

        """

        super(ParticleSwarm, self).__init__(config)

        # Set default values for non-essential parameters - no longer here; now done in Config.

        conf_dict = config.config  # Dictionary from the Configuration object

        # This default value gets special treatment because if missing, it should take the value of particle_weight,
        # disabling the adaptive weight change entirely.
        if 'particle_weight_final' not in conf_dict:
            conf_dict['particle_weight_final'] = conf_dict['particle_weight']

        # Save config parameters
        self.c1 = conf_dict['cognitive']
        self.c2 = conf_dict['social']
        self.max_evals = conf_dict['population_size'] * conf_dict['max_iterations']

        self.num_particles = conf_dict['population_size']
        # Todo: Nice error message if a required key is missing

        self.w0 = conf_dict['particle_weight']

        self.wf = conf_dict['particle_weight_final']
        self.nmax = conf_dict['adaptive_n_max']
        self.n_stop = conf_dict['adaptive_n_stop']
        self.absolute_tol = conf_dict['adaptive_abs_tol']
        self.relative_tol = conf_dict['adaptive_rel_tol']

        self.nv = 0  # Counter that controls the current weight. Counts number of "unproductive" iterations.
        self.num_evals = 0  # Counter for the total number of results received

        # Initialize storage for the swarm data
        self.swarm = []  # List of lists of the form [PSet, velocity]. Velocity is stored as a dict with the same keys
        # as PSet
        self.pset_map = dict()  # Maps each PSet to it s particle number, for easy lookup.
        self.bests = [[None, np.inf]] * self.num_particles  # The best result for each particle: list of the
        # form [PSet, objective]
        self.global_best = [None, np.inf]  # The best result for the whole swarm
        self.last_best = np.inf

    def start_run(self):
        """
        Start the run by initializing n particles at random positions and velocities
        :return:
        """

        for i in range(self.num_particles):
            new_params = self.random_pset()
            # Todo: Smart way to initialize velocity?
            new_velocity = {xi: np.random.uniform(-1, 1) for xi in self.variables}
            self.swarm.append([new_params, new_velocity])
            self.pset_map[new_params] = i

        return [particle[0] for particle in self.swarm]

    def got_result(self, res):
        """
        Updates particle velocity and position after a simulation completes.

        :param res: Result object containing the run PSet and the resulting Data.
        :return:
        """

        paramset = res.pset
        score = res.score

        self.num_evals += 1

        if self.num_evals % self.num_particles == 0:
            # End of one "pseudoflight", check if it was productive.
            if (self.last_best != np.inf and
                    np.abs(self.last_best - self.global_best[1]) <
                    self.absolute_tol + self.relative_tol * self.last_best):
                self.nv += 1
            self.last_best = self.global_best[1]

        p = self.pset_map.pop(paramset)  # Particle number

        # Update best scores if needed.
        if score < self.bests[p][1]:
            self.bests[p] = [paramset, score]
            if score < self.global_best[1]:
                self.global_best = [paramset, score]

        # Update own position and velocity
        # The order matters - updating velocity first seems to make the best use of our current info.
        w = self.w0 + (self.wf - self.w0) * self.nv / (self.nv + self.nmax)
        self.swarm[p][1] = {v:
                                w * self.swarm[p][1][v] + self.c1 * np.random.random() * (
                                self.diff(self.bests[p][0], self.swarm[p][0], v)) +
                                self.c2 * np.random.random() * self.diff(self.global_best[0], self.swarm[p][0], v)
                            for v in self.variables}
        new_pset = PSet({v: self.add(self.swarm[p][0], v, self.swarm[p][1][v]) for v in self.variables})
        self.swarm[p][0] = new_pset

        # This will cause a crash if new_pset happens to be the same as an already running pset in pset_map.
        # This could come up in practice if all parameters have hit a box constraint.
        # As a simple workaround, perturb the parameters slightly
        while new_pset in self.pset_map:
            retry_dict = {v: self.add(new_pset, v, np.random.uniform(-1e-6, 1e-6)) for v in self.variables}
            new_pset = PSet(retry_dict)

        self.pset_map[new_pset] = p

        # Check for stopping criteria
        if self.num_evals >= self.max_evals or self.nv >= self.n_stop:
            return 'STOP'

        return [new_pset]
