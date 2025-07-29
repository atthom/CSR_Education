from alphai import *

connect_wifi()
# Carre vers la gauche
motor(30,30,0.8)
motor(-30,30,0.45)
motor(30,30,0.8)
motor(-30,30,0.5)
motor(30,30,0.8)
motor(-30,30,0.65)
motor(30,30,0.8)
motor(-30,30,0.74)
motor(0,0,2)
# # Triangle
motor(30,30,1.3)
motor(-30,30,0.6)
motor(30,30,1.3)
motor(-30,30,0.6)
motor(30,30,1.6)

disconnect()