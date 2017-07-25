from .context import pset
from nose.tools import raises


class TestModel:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.file1 = 'bngl_files/Simple.bngl'
        cls.file2 = 'bngl_files/ParamsEverywhere.bngl'
        cls.file3 = 'bngl_files/Tricky.bngl'

        cls.file1a = 'bngl_files/Simple_Answer.bngl'

        cls.savefile = 'bngl_files/NoseTest_Save.bngl'

        cls.dict1 = {'kase__FREE__': 3.8, 'pase__FREE__': 0.16, 'koff__FREE__': 4.4e-3}
        cls.dict2 = {'kase__FREE__': 3.8, 'pase__FREE__': 0.16, 'wrongname__FREE__': 4.4e-3}

    def test_initialize(self):
        model1 = pset.Model(self.file1)
        assert model1.param_names == ('kase__FREE__', 'koff__FREE__', 'pase__FREE__')

        model2 = pset.Model(self.file2)
        assert model2.param_names == (
            'Ag_tot_1__FREE__', 'kase__FREE__', 'koff__FREE__', 'kon__FREE__', 'pase__FREE__', 't_end__FREE__')

        model3 = pset.Model(self.file3)
        assert model3.param_names == ('__koff2__FREE__', 'kase__FREE__', 'koff__FREE__')

    def test_set_param_set(self):
        model1 = pset.Model(self.file1)
        ps1 = pset.PSet(self.dict1)
        model1.set_param_set(ps1)
        assert model1.param_set['kase__FREE__'] == 3.8

    @raises(ValueError)
    def test_set_param_set_error(self):
        model1 = pset.Model(self.file1)
        ps2 = pset.PSet(self.dict2)
        model1.set_param_set(ps2)

    def test_model_text(self):
        model1 = pset.Model(self.file1)
        ps1 = pset.PSet(self.dict1)
        model1.set_param_set(ps1)

        f_answer = open(self.file1a)  # File containing the correct output for model_text()
        answer = f_answer.read()
        f_answer.close()
        assert model1.model_text() == answer

    def test_model_save(self):
        model1 = pset.Model(self.file1)
        ps1 = pset.PSet(self.dict1)
        model1.set_param_set(ps1)
        model1.save(self.savefile)

        f_myguess = open(self.savefile)
        myguess = f_myguess.read()
        f_myguess.close()

        f_answer = open(self.file1a)  # File containing the correct output for model_text()
        answer = f_answer.read()
        f_answer.close()

        assert myguess == answer
