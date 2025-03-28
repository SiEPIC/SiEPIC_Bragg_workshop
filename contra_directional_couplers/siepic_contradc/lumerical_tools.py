import sys, os, platform

# Lumerical Python API path on system

#%% find lumapi 
import sys, os, platform

mode = None # variable for the Lumerical Python API

# Lumerical Python API path on system
try:
    import lumapi
except:
    cwd = os.getcwd()

    if platform.system() == 'Darwin':
        path_app = '/Applications'
    elif platform.system() == 'Linux':
        path_app = '/opt'
        os.environ["QT_QPA_PLATFORM"] = "xcb" 
    elif platform.system() == 'Windows': 
        path_app = 'C:\\Program Files'
        os.environ['QT_QPA_PLATFORM']= 'windows'
        print('Windows Qt')
    else:
        print('Not a supported OS')
        exit()


    # Application folder paths containing Lumerical
    p = [s for s in os.listdir(path_app) if "lumerical" in s.lower()]
    # check sub-folders for lumapi.py
    import fnmatch
    for dir_path in p:
        search_str = 'lumapi.py'
        matches = []
        for root, dirnames, filenames in os.walk(os.path.join(path_app,dir_path), followlinks=True):
            for filename in fnmatch.filter(filenames, search_str):
                matches.append(root)
        if matches:
            lumapi_path = matches[0]
    if not lumapi_path in sys.path:
        sys.path.append(lumapi_path)
    #    os.chdir(lumapi_path)

    print('Lumerical lumapi.py path: %s' % lumapi_path)

import lumapi

dir_path = os.path.dirname(os.path.realpath(__file__))
print('Simulation project path: %s' % dir_path)


def generate_dat(pol = 'TE', terminate = True):
    mode = lumapi.open('mode')

    # feed polarization into model
    if pol == 'TE':
        lumapi.evalScript(mode,"mode_label = 'TE'; mode_ID = '1';")
    elif pol == 'TM':
        lumapi.evalScript(mode,"mode_label = 'TM'; mode_ID = '2';")

    # run write sparams script
    lumapi.evalScript(mode,"cd('%s');"%dir_path)
    lumapi.evalScript(mode,'write_sparams;')
    
    if terminate == True:
        lumapi.close(mode)
    
    run_INTC()
    return

def run_INTC():
    intc = lumapi.open('interconnect')
    
    svg_file = "contraDC.svg"
    sparam_file = "ContraDC_sparams.dat"
    command ='cd("%s");'%dir_path
    command += 'switchtodesign; new; deleteall; \n'
    command +='addelement("Optical N Port S-Parameter"); createcompound; select("COMPOUND_1");\n'
    command += 'component = "contraDC"; set("name",component); \n' 
    command += 'seticon(component,"%s");\n' %(svg_file)
    command += 'select(component+"::SPAR_1"); set("load from file", true);\n'
    command += 'set("s parameters filename", "%s");\n' % (sparam_file)
    command += 'set("load from file", false);\n'
    command += 'set("passivity", "enforce");\n'
    
    command += 'addport(component, "%s", "Bidirectional", "Optical Signal", "%s",%s);\n' %("Port 1",'Left',0.25)
    command += 'connect(component+"::RELAY_%s", "port", component+"::SPAR_1", "port %s");\n' % (1, 1)
    command += 'addport(component, "%s", "Bidirectional", "Optical Signal", "%s",%s);\n' %("Port 2",'Left',0.75)
    command += 'connect(component+"::RELAY_%s", "port", component+"::SPAR_1", "port %s");\n' % (2, 2)
    command += 'addport(component, "%s", "Bidirectional", "Optical Signal", "%s",%s);\n' %("Port 3",'Right',0.25)
    command += 'connect(component+"::RELAY_%s", "port", component+"::SPAR_1", "port %s");\n' % (3, 3)
    command += 'addport(component, "%s", "Bidirectional", "Optical Signal", "%s",%s);\n' %("Port 4",'Right',0.75)
    command += 'connect(component+"::RELAY_%s", "port", component+"::SPAR_1", "port %s");\n' % (4, 4)
    command += 'addtolibrary;\n'
    
    lumapi.evalScript(intc, command)
    
    return


def get_DesignKit(name):
    '''
    Query Lumerical INTERCONNECT to find the path for the specific design kit
    '''
    intc = lumapi.open('interconnect')

    # Get all the library elements
    command = 'elements=library;'
    lumapi.evalScript(intc, command)
    intc_elements = lumapi.getVar(intc, "elements")
    intc_elements = intc_elements.split('\n')
    # find the ones that match the requested Design Kit name
    j=[i for i in intc_elements if '::design kits::'+name.lower() in i]
    if not j:
        raise Exception('No elements in the Design Kit "%s" found.' % name)
    # find the first element in the root folder
    intc_element=[i for i in j if len(i.split('::'))==4][0]
    # get the path
    command = f'addelement("{intc_element}"); \n'
    command += 'path=get("local path");'
    lumapi.evalScript(intc, command)
    intc_designkit_path = lumapi.getVar(intc, "path")

    return intc_designkit_path

if __name__ == "__main__":
    print ( get_DesignKit('EBeam') )
