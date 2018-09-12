import ccm
log=ccm.log()

from ccm.lib import grid
from ccm.lib.actr import *

import random

mymap="""
################
#             C#
#     D        #
#              #
##            ##
#   DD         #
#        D    D#
#            DD#
################
"""


class MyCell(grid.Cell):
	dirty=False
	chargingsquare=False
	def color(self):
		if self.chargingsquare: return "green"
		elif self.dirty: return 'brown'
		elif self.wall: return 'black'
		else: return 'white'

	def load(self,char):
		if char=='#': self.wall=True
		elif char=='D': self.dirty=True
		elif char=='C': self.chargingsquare=True

class MotorModule(ccm.Model):
	FORWARD_TIME = .1
	TURN_TIME = 0.025
	CLEAN_TIME = 0.025

	def __init__(self):
		ccm.Model.__init__(self)
		self.busy=False

	def turn_left(self, amount=1):
		if self.busy: return
		self.busy=True
		self.action="turning left"
		yield MotorModule.TURN_TIME
		amount *= -1
		self.parent.body.turn(amount)
		self.busy=False

	def turn_right(self, amount=1):
		if self.busy: return
		self.busy=True
		self.action="turning left"
		yield MotorModule.TURN_TIME
		self.parent.body.turn(amount)
		self.busy=False

	def turn_around(self):
		if self.busy: return
		self.busy=True
		self.action="turning around"
		yield MotorModule.TURN_TIME
		self.parent.body.turn_around()
		self.busy=False

	def go_forward(self, dist=1):
		if self.busy: return
		self.busy=True
		self.action="going forward"
		for i in range(dist):
			yield MotorModule.FORWARD_TIME
			self.parent.body.go_forward()
		self.action=None
		self.busy=False

	def go_left(self,dist=1):
		if self.busy: return
		self.busy="True"
		self.action='turning left'
		yield MotorModule.TURN_TIME
		self.parent.body.turn_left()
		self.action="going forward"
		for i in range(dist):
			yield MotorModule.FORWARD_TIME
			self.parent.body.go_forward()
		self.action=None
		self.busy=False

	def go_right(self):
		if self.busy: return
		self.busy=True
		self.action='turning right'
		yield 0.1
		self.parent.body.turn_right()
		self.action='going forward'
		yield MotorModule.FORWARD_TIME
		self.parent.body.go_forward()
		self.action=None
		self.busy=False

	def go_towards(self,x,y):
		if self.busy: return
		self.busy=True
		self.clean_if_dirty()
		self.action='going towards %s %s'%(x,y)
		yield MotorModule.FORWARD_TIME
		self.parent.body.go_towards(x,y)
		self.action=None
		self.busy=False

	def clean_if_dirty(self):
		"Clean cell if dirty"
		if (self.parent.body.cell.dirty):
			self.action="cleaning cell"
			self.clean()

	def clean(self):
		yield MotorModule.CLEAN_TIME
		self.parent.body.cell.dirty=False



class ObstacleModule(ccm.ProductionSystem):
	production_time=0

	def init():
		self.ahead=body.ahead_cell.wall
		self.left=body.left90_cell.wall
		self.right=body.right90_cell.wall
		self.left45=body.left_cell.wall
		self.right45=body.right_cell.wall


	def check_ahead(self='ahead:False',body='ahead_cell.wall:True'):
		self.ahead=True

	def check_left(self='left:False',body='left90_cell.wall:True'):
		self.left=True

	def check_left45(self='left45:False',body='left_cell.wall:True'):
		self.left45=True

	def check_right(self='right:False',body='right90_cell.wall:True'):
		self.right=True

	def check_right45(self='right45:False',body='right_cell.wall:True'):
		self.right45=True

	def check_ahead2(self='ahead:True',body='ahead_cell.wall:False'):
		self.ahead=False

	def check_left2(self='left:True',body='left90_cell.wall:False'):
		self.left=False

	def check_left452(self='left45:True',body='left_cell.wall:False'):
		self.left45=False

	def check_right2(self='right:True',body='right90_cell.wall:False'):
		self.right=False

	def check_right452(self='right45:True',body='right_cell.wall:False'):
		self.right45=False

class CleanSensorModule(ccm.ProductionSystem):
	production_time = 0
	dirty=False

	def found_dirty(self="dirty:False", body="cell.dirty:True"):
		self.dirty=True

	def found_clean(self="dirty:True", body="cell.dirty:False"):
		self.dirty=False


class VacuumAgent(ACTR):
	goal = Buffer()
	body = grid.Body()
	motorInst = MotorModule()
	cleanSensor = CleanSensorModule()
	obstacle = ObstacleModule()

	def init():
		goal.set("rsearch left 1 0 1")
		self.home = None

	#----ROOMBA----#

	def clean_cell(cleanSensor="dirty:True", utility=0.6):
		motorInst.clean()

	def forward_rsearch(goal="rsearch left ?dist ?num_turns ?curr_dist",
						motorInst="busy:False", body="ahead_cell.wall:False"):
		motorInst.go_forward(int(num_turns)) #go further as the number of turn increases to make a spiral pattern
		print(body.ahead_cell.wall)
		curr_dist = str(int(curr_dist) - 1)
		goal.set("rsearch left ?dist ?num_turns ?curr_dist")

	def left_rsearch(goal="rsearch left ?dist ?num_turns 0", motorInst="busy:False",
					utility=0.1): #change direction when it finishes part of the spiral pattern
		motorInst.turn_left(2) #turn left 90 degrees
		num_turns = str(int(num_turns) + 1) #increase number of turns
		goal.set("rsearch left ?dist ?num_turns ?dist") #set the number of spaces to go

	def wall_rsearch(goal="rsearch left ?dist ?num_turns ?curr_dist", obstacle="ahead:True",utility=0.6): #bounce when it hits the wall
		motorInst.go_left(3) #wall tracing
		goal.set("rsearch left 1 2 1") #set number of turns larger than 0 to keep it away from wall

	def wall2_rsearch(goal="rsearch left ?dist ?num_turns ?curr_dist", obstacle="ahead:True",utility=0.6): #bounce when it hits the wall
		motorInst.go_left(5) #wall tracing with different distance
		goal.set("rsearch left 1 1 1") #use different parameters to prevent circling in same area

world=grid.World(MyCell,map=mymap)
agent=VacuumAgent()
agent.home=()
world.add(agent,5,5,dir=0,color="black")

ccm.log_everything(agent)
ccm.display(world)
world.run()

