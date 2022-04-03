import pickle
import os
import subprocess as sp
from tempfile import NamedTemporaryFile


def vaal_sampling_util(cfg, dataObj, debug=True):
    """Utility function which runs VAAL sampling as python subprocess.

    Args:
        cfg: Reference to config yaml.
        dataObj: Reference to the data class.
        debug (bool, optional): Switch for debubg mode. Defaults to True.
    """
    tempArgsFile = NamedTemporaryFile(
        suffix=".pkl",
        prefix="{}_active_sampling_".format(cfg.ACTIVE_LEARNING.SAMPLING_FN),
        delete=True,
    )  # DELETE=False
    tempArgsFile_fname = tempArgsFile.name

    if debug:
        print("tempArgsFile: {}".format(tempArgsFile_fname))
    temp_cfg_workers = cfg.DATA_LOADER.NUM_WORKERS
    temp_cfg_pinmemory = cfg.DATA_LOADER.PIN_MEMORY
    print("cfg.NUM_WORKERS: ", cfg.DATA_LOADER.NUM_WORKERS)
    cfg.DATA_LOADER.NUM_WORKERS = 0
    temp_list = [cfg, dataObj]
    # Saving the objects:
    with open(tempArgsFile_fname, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(temp_list, f)

    # reset to orig worker
    cfg.DATA_LOADER.NUM_WORKERS = temp_cfg_workers
    scriptName = os.path.join(os.getcwd(), "tools", "vaal_main_exit.py")

    if debug:
        print("scriptname: {}".format(scriptName))

    temp_pythonPath = os.popen("which python").readlines()[0].rstrip()
    sp.call((temp_pythonPath, scriptName, tempArgsFile_fname))

    tempArgsFile.close()


def active_sampling(cfg, ensemble_args=[], debug=False):
    """Utility function which runs AL sampling as python subprocess.

    Args:
        cfg: Reference to config yaml.
        ensemble_args: Args used by ensemble based AL sampling.
        debug (bool, optional): Switch for debubg mode. Defaults to True.
    """
    tempArgsFile = NamedTemporaryFile(
        suffix=".pkl", prefix="active_sampling_", delete=True
    )  # DELETE=False
    tempArgsFile_fname = tempArgsFile.name

    if debug:
        print("tempArgsFile: {}".format(tempArgsFile_fname))

    if cfg.TRAIN.DATASET.upper() == "IMAGENET":
        temp_cfg_workers = cfg.DATA_LOADER.NUM_WORKERS
        temp_cfg_pinmemory = cfg.DATA_LOADER.PIN_MEMORY
        print("cfg.NUM_WORKERS: ", cfg.DATA_LOADER.NUM_WORKERS)
        cfg.DATA_LOADER.NUM_WORKERS = 0
    temp_list = [cfg, ensemble_args]
    # Saving the objects:
    with open(tempArgsFile_fname, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(temp_list, f)

    if cfg.TRAIN.DATASET.upper() == "IMAGENET":
        # reset to orig worker
        cfg.DATA_LOADER.NUM_WORKERS = temp_cfg_workers

    scriptName = os.path.join(os.getcwd(), "tools", "al_sampling_exit.py")

    if debug:
        print("scriptname: {}".format(scriptName))

    temp_pythonPath = os.popen("which python").readlines()[0].rstrip()
    sp.call((temp_pythonPath, scriptName, tempArgsFile_fname))

    tempArgsFile.close()


def test_net_subprocess_call(
    temp_out_dir: str, latest_model_path: str, debug: bool = False
) -> float:
    """Tests the model.

    Args:
        temp_out_dir (str): Path where both config.yaml and checkpoints for the best model are saved.
        latest_model_path (str): Absolute path to .pyth.
        debug (bool, optional): Switch for debug mode. Defaults to False.

    Returns:
        float: Returns the test accuracy.
    """
    if debug:
        print("=================================")
    if debug:
        print("Started test net subprocess call")

    ##test model via subprocess
    testScriptName = os.path.join(os.getcwd(), "tools", "test_net.py")
    testscriptsArgs1 = f"{temp_out_dir}config.yaml"
    testscriptsArgs2 = f"{latest_model_path}"

    print("Subprocess called as : \n")
    temp_pythonPath = os.popen("which python").readlines()[0].rstrip()

    print(
        temp_pythonPath,
        testScriptName,
        "--cfg",
        testscriptsArgs1,
        "TEST.WEIGHTS",
        testscriptsArgs2,
    )

    op_string = sp.check_output(
        (
            temp_pythonPath,
            testScriptName,
            "--cfg",
            testscriptsArgs1,
            "TEST.WEIGHTS",
            testscriptsArgs2,
        )
    )  # , "OUT_DIR", "/tmp"))

    # as check_output returns bytes not string -- so we decode it
    op_string = op_string.decode("ASCII")
    print(op_string)

    #### Extract test accuracy ###
    temp_test_acc = float(op_string.split("[Acquisition:")[1].split("of data is ")[1])
    print("Extracted Test Accuracy from subproces: {}".format(temp_test_acc))
    if debug:
        print(f"Finished test net subprocess call")
    if debug:
        print("=================================")

    return temp_test_acc


def SWA_subprocess_call(argList, debug=False):
    """Utility to run SWA postprocessing as a python suboprocess."""
    if debug:
        print("=================================")
    if debug:
        print("Started SWA subprocess call")

    # Pickle args file into a temp file
    tempArgsFile = NamedTemporaryFile(suffix=".pkl", delete=True)  # DELETE=False
    tempArgsFile_fname = tempArgsFile.name

    if debug:
        print("tempArgsFile: {}".format(tempArgsFile_fname))
    if debug:
        print(
            f"Before calling swa subprocess: cfg.TRAIN.DATASET : {argList[-1].TRAIN.DATASET}"
        )

    # Saving the objects:
    with open(tempArgsFile_fname, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(argList, f)
    scriptName = os.path.join(os.getcwd(), "tools", "train_swa_exit.py")

    if debug:
        print("scriptname: {}".format(scriptName))

    temp_pythonPath = os.popen("which python").readlines()[0].rstrip()
    sp.call((temp_pythonPath, scriptName, tempArgsFile_fname))
    # sp.call (( '/nfs/users/ext_prateek.munjal/anaconda3/envs/pycls/bin/python3.6', scriptName, tempArgsFile_fname ))

    tempArgsFile.close()

    if debug:
        print("Finished SWA subprocess call")
    if debug:
        print("=================================")


def auto_ml_sp(cfg, args, debug=False):
    """Utility function which runs AutoML hyper-parameter tuning as python subprocess.

    Args:
        cfg: Reference to config yaml.
        args: Reference to the input args.
        debug (bool, optional): Switch for debubg mode. Defaults to True.
    """
    tempArgsFile = NamedTemporaryFile(
        suffix=".pkl", prefix="auto_ml_sp_", delete=True
    )  # DELETE=False
    tempArgsFile_fname = tempArgsFile.name

    if debug:
        print("tempArgsFile: {}".format(tempArgsFile_fname))

    temp_list = [cfg, args]
    # Saving the objects:
    with open(tempArgsFile_fname, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(temp_list, f)
    scriptName = os.path.join(os.getcwd(), "tools", "auto_ml_exit.py")

    if debug:
        print("scriptname: {}".format(scriptName))

    temp_pythonPath = os.popen("which python").readlines()[0].rstrip()
    sp.call((temp_pythonPath, scriptName, tempArgsFile_fname))

    tempArgsFile.close()


def swa_on_auto_ml_sp(cfg, args, debug=False):
    """Utility function which runs SWA post-training on models saved during AutoML trials.
    This is implemented as a python subprocess.

    Args:
        cfg: Reference to config yaml.
        args: Reference to the input args.
        debug (bool, optional): Switch for debubg mode. Defaults to True.
    """
    tempArgsFile = NamedTemporaryFile(
        suffix=".pkl", prefix="sp_swa_", delete=True
    )  # DELETE=False
    tempArgsFile_fname = tempArgsFile.name

    if debug:
        print("tempArgsFile: {}".format(tempArgsFile_fname))

    temp_list = [cfg, args]
    # Saving the objects:
    with open(tempArgsFile_fname, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(temp_list, f)

    scriptName = os.path.join(os.getcwd(), "tools", "swa_postrun.py")

    if debug:
        print("scriptname: {}".format(scriptName))

    temp_pythonPath = os.popen("which python").readlines()[0].rstrip()
    sp.call((temp_pythonPath, scriptName, tempArgsFile_fname))

    tempArgsFile.close()
