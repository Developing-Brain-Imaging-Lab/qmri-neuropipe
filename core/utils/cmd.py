import os, sys, subprocess, traceback

def run_cmd(cmd, log=None):
    try:
        process = subprocess.run([cmd], shell=True, capture_output=True, text=True)
        
        if log:
            with open(log,'w') as f_obj:
                f_obj.write(process.stdout)
                f_obj.write("\n")

    except Exception as ex:
        trace = []
        tb = ex.__traceback__
        while tb is not None:
            trace.append({
                "filename": tb.tb_frame.f_code.co_filename,
                "name": tb.tb_frame.f_code.co_name,
                "lineno": tb.tb_lineno
            })
            tb = tb.tb_next
        print(str({
            'type': type(ex).__name__,
            'message': str(ex),
            'trace': trace
        }))
   



