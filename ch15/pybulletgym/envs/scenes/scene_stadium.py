import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

from .scene_bases import Scene
import pybullet as p


class StadiumScene(Scene):
	multiplayer = False
	zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
	stadium_halflen   = 105*0.25	# FOOTBALL_FIELD_HALFLEN
	stadium_halfwidth = 50*0.25	 # FOOTBALL_FIELD_HALFWID

	def episode_restart(self):
		Scene.episode_restart(self)   # contains cpp_world.clean_everything()
		filename = os.path.join(os.path.dirname(__file__), "..", "assets", "scenes", "stadium", "stadium.sdf")
		print(filename)		
		self.stadium = p.loadSDF(filename)
		planeName = os.path.join(os.path.dirname(__file__), "..", "assets", "mjcf/ground_plane.xml")
		
		self.ground_plane_mjcf = p.loadMJCF(planeName)
		for i in self.ground_plane_mjcf:
			p.changeVisualShape(i,-1,rgbaColor=[0,0,0,0])