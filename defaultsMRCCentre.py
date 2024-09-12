import os

def get_subject_id_from_DIR(DIR):
    DIR = DIR.split('^')  # e.g. thigh^data/brcalskd/BRCALSKD_056C
    assert (len(DIR) >= 2)
    DIR = DIR[1]

    if 'arimoclomol' in DIR: 
        subject = os.path.basename(DIR) # e.g. kansas_11_011_12m
        if subject.startswith('kansas'):
            assert (len(subject) >= 13)
            subject = 'arimoclomol.'+subject[:13]
            #print(subject)
            return subject
        elif subject.startswith('nhnn'): # e.g. nhnn_020-031_20m
            assert (len(subject) >= 12)
            subject = 'arimoclomol.'+subject[:12]
            #print(subject)
            return subject
        else:
            assert(False)
                
    if 'dhmn' in DIR: 
        subject = os.path.basename(DIR) # e.g. gait_110b
        assert (len(subject) >= 8)
        subject = 'dhmn.'+subject[:8]
        #print(subject)
        return subject
        
    if 'mdacmt' in DIR: 
        subject = os.path.basename(DIR) # e.g. iowa_066a
        assert (len(subject) == 9)
        subject = 'mdacmt.'+subject[:8]
        #print(subject)
        return subject
        
    if 'alscs' in DIR: 
        subject = os.path.basename(DIR).replace('Baloh_', '') # e.g. Baloh_42350_001B
        assert (len(subject) >= 9)
        subject = 'alscs.'+subject[:9]
        #print(subject)
        return subject
        
    if 'poems' in DIR: 
        subject = os.path.basename(DIR) # e.g. POEMSMRI01
        subject = 'poems.'+subject
        #print(subject)
        return subject
        
    if 'BRCALSKD' in DIR:
        subject = os.path.basename(DIR).replace('BRCALSKD_', '')
        assert (len(subject) == 4)
        return 'BRCALSKD.'+subject[:3]

    if 'HYPOPP' in DIR.upper():
        subject = os.path.basename(DIR)
        assert (len(subject) == 5)
        assert (subject[3] == '_')
        return 'HYPOPP.'+subject[:3]

    if 'ibmcmt_p1' in DIR:
        subject = os.path.basename(DIR).replace('p1-', '')
        assert (len(subject) == 4)
        return 'ibmcmt_p1.'+subject[:3]

    if 'ibmcmt_p2' in DIR:
        subject = os.path.basename(DIR).replace('p2-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    if 'ibmcmt_p3' in DIR:
        subject = os.path.basename(DIR).replace('p3-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    if 'ibmcmt_p4' in DIR:
        subject = os.path.basename(DIR).replace('p4-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    if 'ibmcmt_p5' in DIR:
        subject = os.path.basename(DIR).replace('p5-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    if 'ibmcmt_p6' in DIR:
        subject = os.path.basename(DIR).replace('p6-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    raise Exception('Unable to determine subject ID')
