"""Multiple phantom imports."""
import aioreactor
import blazeutils

def run(task):
    return blazeutils.execute(aioreactor.spawn(task))
