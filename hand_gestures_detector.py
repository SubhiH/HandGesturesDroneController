import matplotlib
matplotlib.use("TkAgg")
from utils import detector_utils as detector_utils
import tkinter as tk
from ScrolledText import ScrolledText
import cv2
import tensorflow as tf
import threading
import datetime
import argparse
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageTk
import matplotlib.animation as animation
import numpy as np
import Queue
from autopilot.autopilot import autopilot



detection_graph, sess = detector_utils.load_inference_graph()

#Control Command
control_command = {}


class hand_gesture_detector:

	def __init__(self,video_streaming_obj):

		####################################
		########GLOBAL VARIABLES############
		####################################
		global control_command
		control_command = {}
		control_command['ARM_TAKEOFF']=0
		control_command['MOVE']=1
		control_command['FLYTING_RIGHT']=2
		control_command['FLYTING_LEFT']=3
		control_command['FLYTING_BACK']=4
		control_command['LAND_DISARM']=5

		####################################
		###########Streaming################
		####################################
		self.video_streaming_obj = video_streaming_obj
		self.frame = None
		self.streaming_thread = None
		self.stopEvent = threading.Event()

		self.streaming_thread = threading.Thread(target=self.videoLoop, args=())
		self.streaming_thread.start()
		####################################
		####################################

		####################################
		###########Autpilot#################
		####################################
		self.autopilot_thread = None
		self.autopilot_obj = None
		self.is_connected_to_autopilot = False
		self.autopilot_sending_msgs_stack = []
		self.autopilot_move_x_y_stack = []
		self.autopilot_speed_shift = []
		self.autopilot_log = []

		####################################
		####################################


		####################################
		########initialize UI window########
		####################################
		self.root = tk.Tk()

		self.panel = None
		self.image = None

		self.panel = tk.Label(image=self.image)
		self.panel.image = self.image
		self.panel.grid(row=0,column=3,rowspan=3,sticky=tk.NSEW)

		self.ip_lbl = tk.Label( self.root, text='IP',justify=tk.LEFT).grid(row=0,column=0,sticky=tk.NW,padx=5)
		self.port_lbl = tk.Label( self.root, text='Port',justify=tk.LEFT).grid(row=0,column=1,sticky=tk.NW)

		self.ip_entry_text = tk.StringVar()
		self.ip_entry = tk.Entry(self.root,width=10,textvariable=self.ip_entry_text)
		self.ip_entry_text.set("127.0.0.1")
		self.ip_entry.grid(row=1,column=0,sticky=tk.NW,padx=5)
		self.port_entry_text = tk.StringVar()
		self.port_entry = tk.Entry(self.root,width=5,textvariable=self.port_entry_text)
		self.port_entry_text.set("14559")
		self.port_entry.grid(row=1,column=1,sticky=tk.NW)
		self.connect_btn = tk.Button(self.root, text ="Connect", command = self.connect_to_autopilot).grid(row=1,column=2,sticky=tk.NW,padx=5)

		self.scrolled_text= ScrolledText(self.root, wrap=tk.WORD,width=40,bg='black')
		self.scrolled_text.grid(row=2,column=0,columnspan=3)
		self.scrolled_text.tag_config('normal', foreground='white')
		self.scrolled_text.tag_config('telemetry', foreground='green')
		self.scrolled_text.tag_config('error', foreground='red')

		# set a callback to handle when the window is closed
		self.root.wm_title("Hand Gestures Detector")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
		####################################
		####################################

		####################################
		#########Detection Variables########
		####################################

		self.log = []

		# max number of hands we want to detect/track
		self.num_hands_detect = 2

		#

		self.prev_first_sample_points_xy = [(0,0),(0,0),(0,0),(0,0),(0,0)]
		self.first_sample_points_xy = [(0,0),(0,0),(0,0),(0,0),(0,0)]

		self.prev_second_sample_points_xy = [(0,0),(0,0),(0,0),(0,0),(0,0)]
		self.second_sample_points_xy = [(0,0),(0,0),(0,0),(0,0),(0,0)]

		im_width, im_height = (int(self.video_streaming_obj.get(3)), int(self.video_streaming_obj.get(4)))
		self.start_x = int(im_width)
		self.start_y = int(im_height)

		self.prev_first_hand_shape = -1
		self.prev_second_hand_shape = -1

		self.prev_box_1 = None
		self.prev_box_2 = None

		self.box_1 = None
		self.box_2 = None

		self.first_hand_shape = -1
		self.second_hand_shape = -1

		self.gestures_queue_first = Queue.Queue()
		self.gestures_queue_second = Queue.Queue()

		self.is_connected = False
		self.arm_pattern = [1, 0, 1]
		self.backward_forward_pattern = [1, 0, 1]
		self.score_thresh = 0.7

		self.output_img = np.zeros((700,1200,3),dtype= np.uint8)


		self.num_of_frames_without_hands = 0
		self.same_hand_shape_counter = 0
		self.num_of_frames_before_flip_hand_boxes = 0
		self.bg_frame = None

		self.lock_wheel = False
		self.num_of_frames_lock_wheel=0

		self.is_moving_forward = True
		self.change_moving_counter = 0

		self.arrow_shift = 0

		#ini_magic_re
		for _ in range(3):
			self.gestures_queue_first.put(-1)
			self.gestures_queue_second.put(-1)
		####################################
		####################################




	def connect_to_autopilot(self):
		if not self.ip_entry.get()=="" and not self.port_entry.get()=="":
			if self.is_connected_to_autopilot:
				self.scrolled_text.insert(tk.END, "Already Connected to Vehcile! \n", 'error')
			else:
				self.autopilot_thread = threading.Thread(target=self.handle_autopilot, args=())
				self.autopilot_thread.start()
		else:
			self.scrolled_text.insert(tk.END, "Enter IP:Port \n", 'error')


	def handle_autopilot(self):
		if not self.is_connected_to_autopilot:
			self.autopilot_obj = autopilot()
			self.autopilot_obj.connect(self.ip_entry.get(),int(self.port_entry.get()))
			if not self.autopilot_obj is None:
				self.is_connected_to_autopilot = True
			#just for test
			# self.is_connected_to_autopilot = True
			# #

		while self.is_connected_to_autopilot:

			incoming_msg = self.autopilot_obj.pop_from_feedback_stack()
			if not incoming_msg is None:
				self.scrolled_text.insert(tk.END, incoming_msg+"\n", 'telemetry')

			if len(self.autopilot_sending_msgs_stack)>0:
				global control_command
				command = self.autopilot_sending_msgs_stack.pop()
				if command == control_command['ARM_TAKEOFF']:
					self.autopilot_obj.change_flight_mode('guided')
					self.autopilot_obj.arm()
					self.autopilot_obj.takeoff(1)
				elif command == control_command['MOVE']:
					if len(self.autopilot_move_x_y_stack)>0:
						(x,y,z)=self.autopilot_move_x_y_stack.pop()
						self.autopilot_obj.move(x/2,y/2,z/2,1)
						print 'move ',x,y,z
				self.scrolled_text.insert(tk.END, self.autopilot_log.pop()+"\n", 'normal')



	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.video_streaming_obj.release()
		self.root.quit()

	def videoLoop(self):
		im_width, im_height = (int(self.video_streaming_obj.get(3)), int(self.video_streaming_obj.get(4)))


		try:
			while not self.stopEvent.is_set():
				_, image_np = self.video_streaming_obj.read()
				image_np = cv2.flip(image_np, 1 )

				# image_np = cv2.flip(image_np, 1)
				try:
					image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
				except:
					print("Error converting to RGB")

				boxes, scores, classes = detector_utils.detect_objects(
					image_np, detection_graph, sess)
				self.image = cv2.cvtColor(self.output_img,cv2.COLOR_BGR2RGB)
				self.image = Image.fromarray(self.image)
				self.image = ImageTk.PhotoImage(self.image)
				self.panel.configure(image=self.image)
				self.panel.image = self.image

				#filter by score
				tmp_scores = []
				tmp_classes = []
				tmp_boxes = []

				for i in range(self.num_hands_detect):
				   if (scores[i] > self.score_thresh):
					   tmp_scores.append(scores[i])
					   tmp_classes.append(classes[i])
					   tmp_boxes.append(boxes[i])

				#filter by score
				filtered_scores = []
				filtered_classes = []
				filtered_boxes = []
				# image_np=detector_utils.draw_left_arrow(image_np)
				for i in range(len(tmp_scores)):
					redundant = False
					(left_1, right_1, top_1, bottom_1) = (tmp_boxes[i][1] * im_width, tmp_boxes[i][3] * im_width,
													  tmp_boxes[i][0] * im_height, tmp_boxes[i][2] * im_height)
					area_1 = (right_1-left_1)*(bottom_1-top_1)
					for j in range(i+1,len(tmp_scores)):
						(left_2, right_2, top_2, bottom_2) = (tmp_boxes[j][1] * im_width, tmp_boxes[j][3] * im_width,
													  tmp_boxes[j][0] * im_height, tmp_boxes[j][2] * im_height)
						area_2 = (right_2-left_2)*(bottom_2-top_2)
						x = max(left_1, left_2)
						y = max(top_1, top_2)
						w = min(right_1, right_2) - x
						h = min(bottom_1, bottom_2) - y
						if w<0 or h<0:
							continue
						else:
							print 'There is intersection'
							if w*h> 0.8*area_1:
								print 'redundant'
								self.log.insert(0,"Remove redundant detection!")
								redundant = True
								break
					if not redundant:
						filtered_scores.append(tmp_scores[i])
						filtered_classes.append(tmp_classes[i])
						filtered_boxes.append(tmp_boxes[i])

				##If No hands appeared for 3 frames, reset the pattern Queues
				if len(filtered_scores)==0:
					self.num_of_frames_without_hands+=1
					print 'No Hands...'
				else:
					self.num_of_frames_without_hands=0

				if self.num_of_frames_without_hands >3:
					self.gestures_queue_second.queue.clear()
					self.gestures_queue_first.queue.clear()
					print 'Reset Patterns...'
					for _ in range(3):
						self.gestures_queue_second.put(-1)
						self.gestures_queue_first.put(-1)

				if self.arrow_shift>3:
					self.arrow_shift = 0
				else:
					self.arrow_shift+=1

				# Lock wheel for 3 frames in case of noise
				if self.lock_wheel and self.num_of_frames_lock_wheel<3:
					if len(filtered_scores)==2 and ((filtered_classes[0]==6.0 and not filtered_classes[1]==6.0) or (not filtered_classes[0]==6.0 and filtered_classes[1]==6.0)):
						print 'LOCK 2 HAND...'
						self.num_of_frames_lock_wheel+=1
						if self.num_of_frames_lock_wheel>=3:
							self.lock_wheel = False
							self.num_of_frames_lock_wheel=0
						image_np = detector_utils.draw_steering_wheel(image_np,self.first_sample_points_xy[0][1]-self.second_sample_points_xy[0][1])
						if self.is_moving_forward:
							cv2.putText(image_np, 'Forward',(int(image_np.shape[1])-65, int(image_np.shape[0])-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0))
						else:
							cv2.putText(image_np, 'Backward',(int(image_np.shape[1])-65, int(image_np.shape[0])-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0))
						# if self.first_sample_points_xy[0][0]>self.second_sample_points_xy[0][0]:
						# 	image_np = detector_utils.draw_steering_wheel(image_np,self.first_sample_points_xy[0][1]-self.second_sample_points_xy[0][1])
						# else:
						# 	image_np = detector_utils.draw_steering_wheel(image_np,self.second_sample_points_xy[0][1]-self.first_sample_points_xy[0][1])
					elif len(filtered_scores)==1 and filtered_classes[0]==6.0:
							image_np = detector_utils.draw_steering_wheel(image_np,0)
							print 'LOCK 1 HAND...'
							self.num_of_frames_lock_wheel+=1
							if self.num_of_frames_lock_wheel>=3:
								self.lock_wheel = False
								self.num_of_frames_lock_wheel=0




				if len(filtered_scores)==1:
					(left_1, right_1, top_1, bottom_1) = (filtered_boxes[0][1] * im_width, filtered_boxes[0][3] * im_width,
											filtered_boxes[0][0] * im_height, filtered_boxes[0][2] * im_height)

					width_1 = right_1 - left_1
					height_1 = bottom_1 - top_1

					self.prev_box_1 = self.box_1
					self.box_1 = filtered_boxes[0]
					self.prev_first_hand_shape = self.first_hand_shape
					self.first_hand_shape = filtered_classes[0]

					self.prev_first_sample_points_xy = self.first_sample_points_xy
					self.first_sample_points_xy = [(int(left_1+width_1/4),int(top_1+height_1/4)),
											(int(right_1-width_1/4),int(top_1+height_1/4)),
											(int(left_1+width_1/4),int(bottom_1-height_1/4)),
											(int(right_1-width_1/4),int(bottom_1-height_1/4)),
											(int(right_1-width_1/2),int(bottom_1-height_1/2))]

					if not list(self.gestures_queue_first.queue)[2] == detector_utils.is_hand_opened(filtered_classes[0]):
							self.gestures_queue_first.get()
							self.gestures_queue_first.put(detector_utils.is_hand_opened(filtered_classes[0]))
							self.same_hand_shape_counter=0
							print list(self.gestures_queue_first.queue)
							if detector_utils.check_pattern(self.gestures_queue_first.queue,self.arm_pattern,self.arm_pattern):
								global control_command
								self.autopilot_sending_msgs_stack.insert(0,control_command['ARM_TAKEOFF'])
								self.autopilot_log.insert(0,"ARM Command is Sent")
								self.is_connected = True
								print("arm sent")
					else:
						self.same_hand_shape_counter+=1

					if self.same_hand_shape_counter >4:
						self.same_hand_shape_counter=0
						print 'Reset Patterns because of latency...'
						self.gestures_queue_first.queue.clear()
						for j in range(3):
							self.gestures_queue_first.put(-1)

					cv2.rectangle(image_np, (int(left_1),int(top_1)), (int(right_1),int(bottom_1)), (0, 0, 255), 1)
					cv2.putText(image_np, 'H1',(int(right_1)-15, int(top_1)-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0))
					cv2.putText(image_np,str(filtered_classes[0]),(int(left_1)-5, int(top_1)-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
					for k in range(5):
						cv2.circle(image_np,self.first_sample_points_xy[k], 2, (0,0,255), -1)
						if not self.prev_first_sample_points_xy[k] == (0,0):
							cv2.line(image_np,self.prev_first_sample_points_xy[k],self.first_sample_points_xy[k],(255,0,0),1)

					self.prev_box_2 = None
					self.box_2 = None
					self.prev_second_hand_shape = None
					self.second_hand_shape = None
					self.gestures_queue_second.queue.clear()
					for j in range(3):
						self.gestures_queue_second.put(-1)

					self.change_moving_counter = 0
					# self.prev_second_sample_points_xy = [(0,0),(0,0),(0,0),(0,0),(0,0)]
					# self.second_sample_points_xy = [(0,0),(0,0),(0,0),(0,0),(0,0)]

				elif len(filtered_scores)==2:
					(left_1, right_1, top_1, bottom_1) = (filtered_boxes[0][1] * im_width, filtered_boxes[0][3] * im_width,
											filtered_boxes[0][0] * im_height, filtered_boxes[0][2] * im_height)

					width_1 = right_1 - left_1
					height_1 = bottom_1 - top_1

					(left_2, right_2, top_2, bottom_2) = (filtered_boxes[1][1] * im_width, filtered_boxes[1][3] * im_width,
											filtered_boxes[1][0] * im_height, filtered_boxes[1][2] * im_height)

					width_2 = right_2 - left_2
					height_2 = bottom_2 - top_2

					coordinates = [[left_1, right_1, top_1, bottom_1,width_1,height_1],[left_2, right_2, top_2, bottom_2,width_2,height_2]]

					left_box_index = 0
					rigth_box_index = 1
					if left_1>left_2:
						left_box_index = 1
						rigth_box_index = 0
					self.prev_box_1 = self.box_1
					self.box_1 = filtered_boxes[left_box_index]
					self.prev_first_hand_shape = self.first_hand_shape
					self.first_hand_shape = filtered_classes[left_box_index]

					self.prev_first_sample_points_xy = self.first_sample_points_xy
											# [(int(left_1+width_1/4),int(top_1+height_1/4)),
											#  (int(right_1-width_1/4),int(top_1+height_1/4)),
											#  (int(left_1+width_1/4),int(bottom_1-height_1/4)),
											#  (int(right_1-width_1/4),int(bottom_1-height_1/4)),
											#  (int(right_1-width_1/2),int(bottom_1-height_1/2))]
					self.first_sample_points_xy = [(int(coordinates[left_box_index][0]+coordinates[left_box_index][4]/4),int(coordinates[left_box_index][2]+coordinates[left_box_index][5]/4)),
													(int(coordinates[left_box_index][1]-coordinates[left_box_index][4]/4),int(coordinates[left_box_index][2]+coordinates[left_box_index][5]/4)),
													(int(coordinates[left_box_index][0]+coordinates[left_box_index][4]/4),int(coordinates[left_box_index][3]-coordinates[left_box_index][5]/4)),
													(int(coordinates[left_box_index][1]-coordinates[left_box_index][4]/4),int(coordinates[left_box_index][3]-coordinates[left_box_index][5]/4)),
													(int(coordinates[left_box_index][1]-coordinates[left_box_index][4]/2),int(coordinates[left_box_index][3]-coordinates[left_box_index][5]/2))]



					if not list(self.gestures_queue_first.queue)[2] == detector_utils.is_hand_opened(filtered_classes[left_box_index]):
						self.gestures_queue_first.get()
						self.gestures_queue_first.put(detector_utils.is_hand_opened(filtered_classes[left_box_index]))
						# print '2 first hand: ',list(self.gestures_queue_first.queue)

					cv2.rectangle(image_np, (int(coordinates[left_box_index][0]),int(coordinates[left_box_index][2])), (int(coordinates[left_box_index][1]),int(coordinates[left_box_index][3])), (0, 0, 255), 1)
					cv2.putText(image_np, 'H1',(int(coordinates[left_box_index][1])-20, int(coordinates[left_box_index][2])-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0))
					cv2.putText(image_np,str(filtered_classes[left_box_index]),(int(coordinates[left_box_index][0])-5, int(coordinates[left_box_index][2])-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))

					self.prev_box_2 = self.box_2
					self.box_2 = filtered_boxes[rigth_box_index]
					self.prev_second_hand_shape = self.second_hand_shape
					self.second_hand_shape = filtered_classes[rigth_box_index]

					self.prev_second_sample_points_xy = self.second_sample_points_xy
					self.second_sample_points_xy = [(int(coordinates[rigth_box_index][0]+coordinates[rigth_box_index][4]/4),int(coordinates[rigth_box_index][2]+coordinates[rigth_box_index][5]/4)),
													(int(coordinates[rigth_box_index][1]-coordinates[rigth_box_index][4]/4),int(coordinates[rigth_box_index][2]+coordinates[rigth_box_index][5]/4)),
													(int(coordinates[rigth_box_index][0]+coordinates[rigth_box_index][4]/4),int(coordinates[rigth_box_index][3]-coordinates[rigth_box_index][5]/4)),
													(int(coordinates[rigth_box_index][1]-coordinates[rigth_box_index][4]/4),int(coordinates[rigth_box_index][3]-coordinates[rigth_box_index][5]/4)),
													(int(coordinates[rigth_box_index][1]-coordinates[rigth_box_index][4]/2),int(coordinates[rigth_box_index][3]-coordinates[rigth_box_index][5]/2))]

					if not list(self.gestures_queue_second.queue)[2] == detector_utils.is_hand_opened(filtered_classes[rigth_box_index]):
						self.gestures_queue_second.get()
						self.gestures_queue_second.put(detector_utils.is_hand_opened(filtered_classes[rigth_box_index]))
						# print '2 second hand: ',list(self.gestures_queue_second.queue)
					if filtered_classes[left_box_index] == 3.0 and filtered_classes[left_box_index]==3.0:
						self.change_moving_counter+=1
					else:
						self.change_moving_counter=0

					if self.change_moving_counter>=6:
						self.change_moving_counter=0
						if self.is_moving_forward:
							self.is_moving_forward = False
						else:
							self.is_moving_forward = True

					#HERE
					'''
					if  list(self.gestures_queue_first.queue)[2] == detector_utils.is_hand_opened(filtered_classes[left_box_index]) and  list(self.gestures_queue_second.queue)[2] == detector_utils.is_hand_opened(filtered_classes[rigth_box_index]):
							self.gestures_queue_first.get()
							self.gestures_queue_first.put(detector_utils.is_hand_opened(filtered_classes[left_box_index]))

							self.gestures_queue_second.get()
							self.gestures_queue_second.put(detector_utils.is_hand_opened(filtered_classes[rigth_box_index]))

							self.same_hand_shape_counter=0
							print 'left: ',list(self.gestures_queue_first.queue)
							print 'right: ',list(self.gestures_queue_second.queue)
							if detector_utils.check_pattern(self.gestures_queue_first.queue,self.arm_pattern,self.arm_pattern) and detector_utils.check_pattern(self.gestures_queue_second.queue,self.arm_pattern,self.arm_pattern):
								global control_command
								self.autopilot_sending_msgs_stack.insert(0,control_command['MOVE'])
								self.autopilot_log.insert(0,"MOVE Command is Sent")
								print("MOVE sent")
					else:
						self.same_hand_shape_counter+=1
						print list(self.gestures_queue_first.queue),list(self.gestures_queue_second.queue),detector_utils.is_hand_opened(filtered_classes[rigth_box_index]),detector_utils.is_hand_opened(filtered_classes[left_box_index])

					if self.same_hand_shape_counter >4:
						self.same_hand_shape_counter=0
						print 'Reset Patterns because of latency...'
						self.gestures_queue_first.queue.clear()
						self.gestures_queue_second.queue.clear()
						for j in range(3):
							self.gestures_queue_first.put(-1)
							self.gestures_queue_second.put(-1)
					'''

					cv2.rectangle(image_np, (int(coordinates[rigth_box_index][0]),int(coordinates[rigth_box_index][2])), (int(coordinates[rigth_box_index][1]),int(coordinates[rigth_box_index][3])), (255, 0, 0), 1)
					cv2.putText(image_np, 'H2',(int(coordinates[rigth_box_index][1])-20, int(coordinates[rigth_box_index][2])-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0))
					cv2.putText(image_np,str(filtered_classes[rigth_box_index]),(int(coordinates[rigth_box_index][0])-5, int(coordinates[rigth_box_index][2])-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))

					#show moving direction
					forward = 1
					if self.is_moving_forward:
						cv2.putText(image_np, 'Forward',(int(image_np.shape[1])-65, int(image_np.shape[0])-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0))
						self.log.insert(0,"Direction Changed to Forward!")
					else:
						forward = -1
						cv2.putText(image_np, 'Backward',(int(image_np.shape[1])-65, int(image_np.shape[0])-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0))
						self.log.insert(0,"Direction Changed to Backward!")


					#show Wheel when shapes: close close
					global control_command
					if  not detector_utils.is_hand_opened(self.first_hand_shape) and not detector_utils.is_hand_opened(self.second_hand_shape):
						self.lock_wheel = True
						self.num_of_frames_lock_wheel=0
						shift = self.first_sample_points_xy[0][1]-self.second_sample_points_xy[0][1]

						if shift<-75:
							self.autopilot_speed_shift.insert(0,(forward*0.5,2,0))
						elif shift>-75 and shift<-50:
							self.autopilot_speed_shift.insert(0,(forward*1,1.5,0))
						elif shift>-50 and shift<-25:
							self.autopilot_speed_shift.insert(0,(forward*1.5,1,0))
						elif shift>-25 and shift<-15:
							self.autopilot_speed_shift.insert(0,(forward*2,0.5,0))
						elif shift>-15 and shift<15:
							self.autopilot_speed_shift.insert(0,(forward*2,0,0))
						elif shift>15 and shift<25:
							self.autopilot_speed_shift.insert(0,(forward*2,-0.5,0))
						elif shift>25 and shift<50:
							self.autopilot_speed_shift.insert(0,(forward*1.5,-1,0))
						elif shift>50 and shift<75:
							self.autopilot_speed_shift.insert(0,(forward*1,-1.5,0))
						elif shift>75:
							self.autopilot_speed_shift.insert(0,(forward*0.5,-2,0))

						if len(self.autopilot_speed_shift)>0:
							while len(self.autopilot_move_x_y_stack)>3:
								self.autopilot_sending_msgs_stack.pop()
								self.autopilot_move_x_y_stack.pop()
								self.autopilot_log.pop()
							self.autopilot_sending_msgs_stack.insert(0,control_command['MOVE'])
							self.autopilot_move_x_y_stack.insert(0,self.autopilot_speed_shift[0])
							self.autopilot_log.insert(0,"MOVE Command is Sent X "+str(self.autopilot_speed_shift[0][0])+" Y "+str(self.autopilot_speed_shift[0][1])+" Z "+str(self.autopilot_speed_shift[0][2]))
							# print("MOVE Command is Sent X "+str(self.autopilot_speed_shift[0][0])+" Y "+str(self.autopilot_speed_shift[0][1]))
						image_np = detector_utils.draw_steering_wheel(image_np,self.first_sample_points_xy[0][1]-self.second_sample_points_xy[0][1])


						# if self.first_sample_points_xy[0][0]>self.second_sample_points_xy[0][0]:
						# 	image_np = detector_utils.draw_steering_wheel(image_np,self.first_sample_points_xy[0][1]-self.second_sample_points_xy[0][1])
						# else:
						# 	image_np = detector_utils.draw_steering_wheel(image_np,self.second_sample_points_xy[0][1]-self.first_sample_points_xy[0][1])

					#show arrow when shapes: open close - Move Right - Left
					if detector_utils.is_hand_opened(self.first_hand_shape)==1 and detector_utils.is_hand_opened(self.second_hand_shape)==0:
						self.lock_wheel = False
						self.autopilot_speed_shift.insert(0,(0,2,0))
						while len(self.autopilot_move_x_y_stack)>3:
								self.autopilot_sending_msgs_stack.pop()
								self.autopilot_move_x_y_stack.pop()
								self.autopilot_log.pop()
						self.autopilot_sending_msgs_stack.insert(0,control_command['MOVE'])
						self.autopilot_move_x_y_stack.insert(0,self.autopilot_speed_shift[0])
						self.autopilot_log.insert(0,"MOVE RIGHT Command is Sent X "+str(self.autopilot_speed_shift[0][0])+" Y "+str(self.autopilot_speed_shift[0][1])+" Z "+str(self.autopilot_speed_shift[0][2]))
						image_np = detector_utils.draw_right_arrow(image_np,self.arrow_shift)
					elif  detector_utils.is_hand_opened(self.first_hand_shape)==0 and  detector_utils.is_hand_opened(self.second_hand_shape)==1:
						self.lock_wheel = False
						self.autopilot_speed_shift.insert(0,(0,-2,0))
						while len(self.autopilot_move_x_y_stack)>3:
								self.autopilot_sending_msgs_stack.pop()
								self.autopilot_move_x_y_stack.pop()
								self.autopilot_log.pop()
						self.autopilot_sending_msgs_stack.insert(0,control_command['MOVE'])
						self.autopilot_move_x_y_stack.insert(0,self.autopilot_speed_shift[0])
						self.autopilot_log.insert(0,"MOVE LEFT Command is Sent X "+str(self.autopilot_speed_shift[0][0])+" Y "+str(self.autopilot_speed_shift[0][1])+" Z "+str(self.autopilot_speed_shift[0][2]))
						image_np = detector_utils.draw_left_arrow(image_np,self.arrow_shift)
					elif detector_utils.is_hand_opened(self.first_hand_shape)==0 and detector_utils.is_hand_opened(self.second_hand_shape)==-1:
						self.lock_wheel = False
						self.autopilot_speed_shift.insert(0,(0,0,1))
						while len(self.autopilot_move_x_y_stack)>3:
								self.autopilot_sending_msgs_stack.pop()
								self.autopilot_move_x_y_stack.pop()
								self.autopilot_log.pop()
						self.autopilot_sending_msgs_stack.insert(0,control_command['MOVE'])
						self.autopilot_move_x_y_stack.insert(0,self.autopilot_speed_shift[0])
						self.autopilot_log.insert(0,"MOVE DOWN Command is Sent X "+str(self.autopilot_speed_shift[0][0])+" Y "+str(self.autopilot_speed_shift[0][1])+" Z "+str(self.autopilot_speed_shift[0][2]))
						image_np = detector_utils.draw_down_arrow(image_np,self.arrow_shift)
					elif detector_utils.is_hand_opened(self.first_hand_shape)==-1 and detector_utils.is_hand_opened(self.second_hand_shape)==0:
						self.lock_wheel = False
						self.autopilot_speed_shift.insert(0,(0,0,-1))
						while len(self.autopilot_move_x_y_stack)>3:
								self.autopilot_sending_msgs_stack.pop()
								self.autopilot_move_x_y_stack.pop()
								self.autopilot_log.pop()
						self.autopilot_sending_msgs_stack.insert(0,control_command['MOVE'])
						self.autopilot_move_x_y_stack.insert(0,self.autopilot_speed_shift[0])
						self.autopilot_log.insert(0,"MOVE UP Command is Sent X "+str(self.autopilot_speed_shift[0][0])+" Y "+str(self.autopilot_speed_shift[0][1])+" Z "+str(self.autopilot_speed_shift[0][2]))
						image_np = detector_utils.draw_up_arrow(image_np,self.arrow_shift)
					elif detector_utils.is_hand_opened(self.first_hand_shape)==1 and detector_utils.is_hand_opened(self.second_hand_shape)==1:
						# Clear movement
						self.lock_wheel = False
						self.autopilot_speed_shift.insert(0,(0,0,0))
						while len(self.autopilot_move_x_y_stack)>3:
								self.autopilot_sending_msgs_stack.pop()
								self.autopilot_move_x_y_stack.pop()
								self.autopilot_log.pop()
						self.autopilot_sending_msgs_stack.insert(0,control_command['MOVE'])
						self.autopilot_move_x_y_stack.insert(0,self.autopilot_speed_shift[0])
						self.autopilot_log.insert(0,"PAUSE MOVEMENT Command is Sent X "+str(self.autopilot_speed_shift[0][0])+" Y "+str(self.autopilot_speed_shift[0][1])+" Z "+str(self.autopilot_speed_shift[0][2]))



					#show sample points for each detected hand
					for k in range(5):
						cv2.circle(image_np,self.first_sample_points_xy[k], 2, (0,0,255), -1)
						if not self.prev_first_sample_points_xy[k] == (0,0):
							cv2.line(image_np,self.prev_first_sample_points_xy[k],self.first_sample_points_xy[k],(255,0,0),1)

						cv2.circle(image_np,self.second_sample_points_xy[k], 2, (0,0,255), -1)
						if not self.prev_second_sample_points_xy[k] == (0,0):
							cv2.line(image_np,self.prev_second_sample_points_xy[k],self.second_sample_points_xy[k],(255,0,0),1)
				# else:
				# 	print 'No HANDS *_*', len(filtered_boxes)

				# image_np = detector_utils.draw_steering_wheel(image_np,50)
				image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
				self.output_img=image_np#[0:image_np.shape[0],0:image_np.shape[1],:]=image_np
				# now = datetime.datetime.now()
				# cv2.imwrite('/Users/Soubhi/Desktop/results/'+str(now.second)+'.png',image_np)

		except RuntimeError, e:
			print("[INFO] caught a RuntimeError",str(e))


if __name__ == '__main__':
	video_stream = cv2.VideoCapture(0)
	video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
	video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

	start_time = datetime.datetime.now()
	num_frames = 0


	# start the app
	hgd = hand_gesture_detector(video_stream)
	hgd.root.mainloop()
