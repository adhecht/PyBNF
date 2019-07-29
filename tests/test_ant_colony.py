from .context import data, algorithms, pset, objective, config, parse
import numpy.testing as npt
from os import mkdir, path
from shutil import rmtree
from copy import deepcopy

class TestAntColony:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.exp_data = [
            '# time    v1_result    v2_result    v3_result  v1_result_SD  v2_result_SD  v3_result_SD\n',
            ' 0 3   4   5   0.1   0.2   0.3\n',
            ' 1 2   3   6   0.1   0.1   0.1\n',
            ' 2 4   2   10  0.3   0.1   1.0\n'
        ]
        cls.exp_data_obj = data.Data()
        cls.exp_data_obj.data = cls.exp_data_obj._read_file_lines(cls.exp_data, '\s+')

        cls.sim_data = cls.data1s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.1   3.1   6.1\n',
        ]
        cls.sim_data_obj = data.Data()
        cls.sim_data_obj.data = cls.sim_data_obj._read_file_lines(cls.sim_data, '\s+')

        cls.config = config.Configuration({'population_size':100, 'max_iterations':5, 'search_locality':0.4,
                                           'convergence_speed':0.005, 'archive_size':10, 
                                           'fit_type': 'aco', 
                                           'models': {'bngl_files/parabola.bngl'},
                                                      'exp_data': {'bngl_files/par1.exp'},
                                           'initialization': 'lh', 
                                           'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
                                           'output_dir': 'test_aco', 
                                           ('uniform_var', 'v1__FREE'): [0, 10],
                                           ('loguniform_var', 'v2__FREE'): [0.01, 1e5],
                                           ('lognormal_var', 'v3__FREE'): [0, 1]})

    @classmethod
    def teardown_class(cls):
        if path.isdir('test_aco'):
            rmtree('test_aco')

    def test_random_pset(self):
        """Designed to test random parameter set generation"""
        pass

    def test_archive_pset(self):
        """Designed to test parameter set generation from archived solutions"""
        aco = algorithms.AntColony(self.config)


    def test_start(self):
        """Designed to test algorithm initialization"""
        aco = algorithms.AntColony(self.config)

        # Check solution archive set-up
        assert(len(aco.archive) == 0)
        assert(aco.archive_size == 10)

        # Generate initial parameter sets
        start_params = aco.start_run()

        # We should have generated population_size parameter sets
        assert(len(start_params) == aco.population_size)        
        # None of the parameter sets should have been added to the solution archive yet.
        assert(len(aco.archive) == 0)

    def test_archive_update(self):
        """Designed to test updating the solution archive prior to filling the archive"""
        aco = algorithms.AntColony(self.config)
        start_params = aco.start_run()
        next_params = []
        for p in start_params:
            new_result = algorithms.Result(p, self.sim_data_obj, 'sim_1')
            new_result.score = aco.objective.evaluate(self.sim_data_obj, self.exp_data_obj)
            next_params.append(aco.got_result(new_result))
        assert(len(aco.archive) == 10)
        proposed_best = aco.archive[0]['score']
        for solution in aco.archive:
            assert(proposed_best >= solution['score'])


