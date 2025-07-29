
def reward(state):
    print(state.keys())
    print(state['last_action'])
    print(state['in_v_320x240'])
    print(type(state['in_v_320x240']))
    print(state['forward_motion'])
    print(type(state['forward_motion'])) # roue bloauees = 0 , recule valeur negative, avance valeur positive
    return 0