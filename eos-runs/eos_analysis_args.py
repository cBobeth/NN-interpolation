
# Define arguments for EOS analysis of b -> c form factors
# based on work 
#    Bordone/Gubernari/Jung/van Dyk arXiv:1912.09335

analysis_args = {
    'global_options': {
        'model':           'WET',
        'form-factors':    'HQET',
        'z-order-bound':   '2',
        'z-order-lp':      '3',
        'z-order-slp':     '2',
        'z-order-sslp':    '1',
        'SU3F-limit-sslp': '1',
    },
    'likelihood': [
        ## Theory
        # B-LCSR
        'B->D^(*)::FormFactors[f_+,f_0,A_0,A_1,A_2,V,T_1,T_2,T_23]@GKvD:2018A',
        # Lattice B->D
        'B->D::f_++f_0@FNAL+MILC:2015B',
        'B->D::f_++f_0@HPQCD:2015A',
        # Lattice B->D^*
        'B->D^*::A_1[s_max]@HFLAV:2019A',
        # bounds
        'b->c::Bound[0^+]',
        'b->c::Bound[0^-]',
        'b->c::Bound[1^+]',
        'b->c::Bound[1^-]',
        ## Experiment
        #'B^0->(Dpi)^+l^-nubar::KinematicDistributionsAngularObservables@BBGJvD:2021A',
    ],
    'manual_constraints': {
        # QCDSR
        "B(*)->D(*)::chi_2(1)": {
            'type': 'Gaussian',
            'observable': "B(*)->D(*)::chi_2(1)@HQET",
            'kinematics': { },
            'options':    {
                'form-factors':    'HQET',
                'z-order-bound':   '2',
                'z-order-lp':      '3',
                'z-order-slp':     '2',
                'z-order-sslp':    '1',
                'SU3F-limit-sslp': '1',
            },
            'mean':       -0.06,
            'sigma-stat': { 'hi': 0.02, 'lo': 0.02 },
            'sigma-sys':  { 'hi': 0.00, 'lo': 0.00 },
            'dof': 1
        },
        "B(*)->D(*)::chi_2'(1)": {
            'type': 'Gaussian',
            'observable': "B(*)->D(*)::chi_2'(1)@HQET",
            'kinematics': { },
            'options':    {
                'form-factors':    'HQET',
                'z-order-bound':   '2',
                'z-order-lp':      '3',
                'z-order-slp':     '2',
                'z-order-sslp':    '1',
                'SU3F-limit-sslp': '1',
            },
            'mean':       +0.00,
            'sigma-stat': { 'hi': 0.02, 'lo': 0.02 },
            'sigma-sys':  { 'hi': 0.00, 'lo': 0.00 },
            'dof': 1
        },
        "B(*)->D(*)::chi_3'(1)": {
            'type': 'Gaussian',
            'observable': "B(*)->D(*)::chi_3'(1)@HQET",
            'kinematics': { },
            'options':    {
                'form-factors':    'HQET',
                'z-order-bound':   '2',
                'z-order-lp':      '3',
                'z-order-slp':     '2',
                'z-order-sslp':    '1',
                'SU3F-limit-sslp': '1',
            },
            'mean':       +0.04,
            'sigma-stat': { 'hi': 0.02, 'lo': 0.02 },
            'sigma-sys':  { 'hi': 0.00, 'lo': 0.00 },
            'dof': 1
        },
        'B(*)->D(*)::eta(1)': {
            'type': 'Gaussian',
            'observable': "B(*)->D(*)::eta(1)@HQET",
            'kinematics': { },
            'options':    {
                'form-factors':    'HQET',
                'z-order-bound':   '2',
                'z-order-lp':      '3',
                'z-order-slp':     '2',
                'z-order-sslp':    '1',
                'SU3F-limit-sslp': '1',
            },
            'mean':       +0.62,
            'sigma-stat': { 'hi': 0.20, 'lo': 0.20 },
            'sigma-sys':  { 'hi': 0.00, 'lo': 0.00 },
            'dof': 1
        },
        "B(*)->D(*)::eta'(1)": {
            'type': 'Gaussian',
            'observable': "B(*)->D(*)::eta'(1)@HQET",
            'kinematics': { },
            'options':    {
                'form-factors':    'HQET',
                'z-order-bound':   '2',
                'z-order-lp':      '3',
                'z-order-slp':     '2',
                'z-order-sslp':    '1',
                'SU3F-limit-sslp': '1',
            },
            'mean':       +0.00,
            'sigma-stat': { 'hi': 0.20, 'lo': 0.20 },
            'sigma-sys':  { 'hi': 0.00, 'lo': 0.00 },
            'dof': 1
        },
        # Lattice ratio f_t over f_+
        'B->D::f_T/f_+': {
            'type': 'Gaussian',
            'observable': 'B->D::f_T(q2)/f_+(q2)',
            'kinematics': { 'q2': 11.643 },
            'options':    {
                'form-factors':    'HQET',
                'z-order-bound':   '2',
                'z-order-lp':      '3',
                'z-order-slp':     '2',
                'z-order-sslp':    '1',
                'SU3F-limit-sslp': '1',
            },
            'mean':       1.113,
            'sigma-stat': { 'hi': 0.126, 'lo': 0.126 },
            'sigma-sys':  { 'hi': 0.000, 'lo': 0.000 },
            'dof': 1
        }
    },
    'priors': [
        # LP
        { 'parameter':     "B(*)->D(*)::xi'(1)@HQET", 'min':  -2.00,   'max':  -0.2,    'type': 'uniform' },
        { 'parameter':    "B(*)->D(*)::xi''(1)@HQET", 'min':  -0.20,   'max':  +4.0,    'type': 'uniform' },
        { 'parameter':   "B(*)->D(*)::xi'''(1)@HQET", 'min': -10.00,   'max':  +0.5,    'type': 'uniform' },
        #
        # SLP
        { 'parameter':   "B(*)->D(*)::chi_2(1)@HQET", 'min':  -0.26,   'max':  +0.14,   'type': 'uniform' },
        { 'parameter':  "B(*)->D(*)::chi_2'(1)@HQET", 'min':  -0.21,   'max':  +0.19,   'type': 'uniform' },
        { 'parameter': "B(*)->D(*)::chi_2''(1)@HQET", 'min':  -1.20,   'max':  +1.20,   'type': 'uniform' },
        
        { 'parameter':  "B(*)->D(*)::chi_3'(1)@HQET", 'min':  -0.16,   'max':  +0.24,   'type': 'uniform' },
        { 'parameter': "B(*)->D(*)::chi_3''(1)@HQET", 'min':  -0.50,   'max':  +0.30,   'type': 'uniform' },
        
        { 'parameter':     "B(*)->D(*)::eta(1)@HQET", 'min':  -0.39,   'max':  +1.81,   'type': 'uniform' },
        { 'parameter':    "B(*)->D(*)::eta'(1)@HQET", 'min':  -1.76,   'max':  +1.64,   'type': 'uniform' },
        { 'parameter':   "B(*)->D(*)::eta''(1)@HQET", 'min':  -3.00,   'max':  +3.00,   'type': 'uniform' },
        #
        # SSLP
        { 'parameter':     "B(*)->D(*)::l_1(1)@HQET", 'min':  -1.30, 'max':  +1.50, 'type': 'uniform' },
        { 'parameter':    "B(*)->D(*)::l_1'(1)@HQET", 'min': -45.00, 'max': +22.00, 'type': 'uniform' },
        
        { 'parameter':     "B(*)->D(*)::l_2(1)@HQET", 'min':  -4.00, 'max':  +0.00, 'type': 'uniform' },
        { 'parameter':    "B(*)->D(*)::l_2'(1)@HQET", 'min': -35.00, 'max': +25.00, 'type': 'uniform' },
        
        # { 'parameter':     "B(*)->D(*)::l_3(1)@HQET", 'min':  -4.00, 'max': +40.00, 'type': 'uniform' },
        { 'parameter':     "B(*)->D(*)::l_3(1)@HQET", 'min': -20.00, 'max': +20.00, 'type': 'uniform' },
        { 'parameter':    "B(*)->D(*)::l_3'(1)@HQET", 'min': -30.00, 'max': +40.00, 'type': 'uniform' },
        
        { 'parameter':     "B(*)->D(*)::l_4(1)@HQET", 'min': -10.00, 'max':  +7.00, 'type': 'uniform' },
        { 'parameter':    "B(*)->D(*)::l_4'(1)@HQET", 'min': -12.00, 'max': +12.00, 'type': 'uniform' },
        
        { 'parameter':     "B(*)->D(*)::l_5(1)@HQET", 'min': -10.00, 'max': +16.00, 'type': 'uniform' },
        { 'parameter':    "B(*)->D(*)::l_5'(1)@HQET", 'min': -13.00, 'max': +16.00, 'type': 'uniform' },
        
        { 'parameter':     "B(*)->D(*)::l_6(1)@HQET", 'min': -15.00, 'max': +20.00, 'type': 'uniform' },
        { 'parameter':    "B(*)->D(*)::l_6'(1)@HQET", 'min': -20.00, 'max': +20.00, 'type': 'uniform' },
    ]
}