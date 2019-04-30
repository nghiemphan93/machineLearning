#! /usr/bin/env python2

# thanks to mdp for many ideas and suggestions

# -- added RMB, Wheel (24 Mar 2018, MDP)

# https://www.mobileread.com/forums/showthread.php?t=272001

from __future__ import print_function

# some assumptions:
#
#   device connected to host (via USB only?) that has adb installed
#   device has 'ok'ed being connected to by adb (developer options + dialog)
#   only one android device connected
#   host display is using 'mirrored' mode - 'extended' not supported (yet?)

# python requirements
#
#   python 2.x
#   PyUserInput - https://pypi.python.org/pypi/PyUserInput
#   # if profiling desired (and see near end of code):
#   yappi - https://pypi.python.org/pypi/yappi

# add/tweak relevant devices and resolutions to 'device_res' below
#
# to determine the model to use as a key (e.g. Max2) below, try:
#
#   adb shell getprop ro.product.board
#
# with the device connected via USB
device_res = {"Max2": (2200, 1650),
			  "Note": (1872, 1404)}

# https://gist.github.com/scottgwald/6862517
# from http://stackoverflow.com/questions/11524586/accessing-logcat-from-android-via-python
import queue
import subprocess
import threading
from datetime import datetime, timedelta
import time
import re
import sys
from pymouse import PyMouse

id_str = "OnyxMonitEv"

# XXX: simplify if necessary
line_re = re.compile("^\d+-\d+\s+" +
			"(?P<time>\d+:\d+:\d+\.\d+)\s.*?" +
			 id_str + ":\s+" +
			 "(?P<event_type>.):(?P<event_action>\d+)\|" +
			 "(?P<x>\d+),(?P<y>\d+).*")

time_fmt = "%H:%M:%S.%f"

dbl_click_delta = timedelta(milliseconds=700)
dbl_click_res = 50

class AsyncFileReader(threading.Thread):
	'''
	Helper class to implement asynchronous reading of a file
	in a separate thread. Pushes read lines on a queue to
	be consumed in another thread.
	'''

	def __init__(self, fd, queue):
		assert isinstance(queue, Queue.Queue)
		assert callable(fd.readline)
		threading.Thread.__init__(self)
		self._fd = fd
		self._queue = queue

	def run(self):
		'''The body of the thread: read lines and put them on the queue.'''
		for line in iter(self._fd.readline, ''):
			self._queue.put(line)

	def eof(self):
		'''Check whether there is no more content to expect.'''
		return not self.is_alive() and self._queue.empty()


def guess_model():
	return subprocess.check_output(['adb', 'shell', 'getprop', 'ro.product.board']).rstrip()


def get_device_res():
	model = guess_model()
	if model in device_res.keys():
		return device_res[model]
	else:
		#print("unexpected model guess: " + model, file=sys.stderr)
		raise ValueError("Unxpected model: ", model)


def main():
	try:
		mouse = PyMouse()
		(host_w, host_h) = mouse.screen_size()
		(device_w, device_h) = get_device_res()
		
		factor_x = host_w / (device_w * 1.0)
		factor_y = host_h / (device_h * 1.0)

		cmd_list = ['adb', 'logcat',
					'-v', 'threadtime',   # in case the default changes again
					'-T', '1',            # show starting at most recent line
					id_str + ':I', '*:S'] 
		# undocumented: https://stackoverflow.com/a/14837250
		#print(subprocess.list2cmdline(cmd_list), file=sys.stderr)
		process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE)

		stdout_queue = Queue.Queue()
		stdout_reader = AsyncFileReader(process.stdout, stdout_queue)
		stdout_reader.start()

		last_down_tm = datetime.now()
		last_down_x = -99999				# trying to choose unlikely coords
		last_down_y = -99999
		
		secondrelease_x = -99999
		secondrelease_y = -99999
		
		flg_FirstTapDown  = -1				# also acts as timer
		flg_SecondTapDown = -1
		flg_TapRelease    = -1
		
		while not stdout_reader.eof():
			time.sleep(0.02)
			
			if flg_FirstTapDown>0 and (time.time()-flg_FirstTapDown)>0.1:
				mouse.press(x, y)						#LMB Down
				flg_FirstTapDown=-1
			
			if flg_TapRelease>0 and flg_FirstTapDown==-1:
				if flg_SecondTapDown>0:
					distance = last_down_y-secondrelease_y
					if abs(distance)<dbl_click_res:
						mouse.click(x,y,2)				#RMB Click
					elif abs(distance)<10000:
						lines = distance*33/device_h
						mouse.scroll(vertical=lines)	#Wheel
					flg_SecondTapDown=-1
				else:
					mouse.release(x, y)					#LMB Up
				flg_TapRelease=-1
			
			while not stdout_queue.empty():
				line = stdout_queue.get()
				#print(line, file=sys.stderr)
				m = line_re.match(line)
				if m:
					tm = datetime.strptime(m.group("time"), time_fmt)
					e = int(m.group("event_action"))
					x = int(m.group("x")) * factor_x
					y = int(m.group("y")) * factor_y
					print( "%s  |  %3i  | %4i,%4i" % (tm.time(), e, int(x), int(y)) )
					
					if (e == 0):
						if ((tm - last_down_tm) < dbl_click_delta) and \
						(abs(x - last_down_x) < dbl_click_res) and \
						(abs(y - last_down_y) < dbl_click_res):
							x = last_down_x
							y = last_down_y
						last_down_tm = tm
						last_down_x = x
						last_down_y = y
						flg_FirstTapDown = time.time()
					elif (e == 1):
						flg_TapRelease=1
					elif (e == 261):
						if flg_FirstTapDown>0:
							flg_FirstTapDown=-1
							flg_SecondTapDown=1
						secondrelease_x = -99999
						secondrelease_y = -99999
					elif (e == 6):
						secondrelease_x = x
						secondrelease_y = y
					else:
						print("unexpected event action: %i" % ( e ), file=sys.stderr)
						print(line, file=sys.stderr)
			
				
			
	# just to catch the likes of C-c to allow clean up etc.
	except KeyboardInterrupt:
		pass
	except ValueError:
		pass
	finally:
		if 'process' in locals():
			process.kill()

# the yappi stuff below is for profiling

# adapted profiling bits from:
#   https://gist.github.com/kwlzn/42b809558dc825821ddb

# import atexit, yappi

# def init_yappi():
#     yappi.set_clock_type('cpu') # or 'wall' or ...
#     yappi.start() # can pass builtins=True if desired

# def finish_yappi():
#     yappi.stop()
 
#     stats = yappi.get_func_stats()
#     # resulting callgrind.out can be used w/ kcachegrind or friends
#     for stat_type in ['pstat', 'callgrind', 'ystat']:
#         stats.save('{}.out'.format(stat_type), type=stat_type)
 
#     with open('func_stats.out', 'wb') as fh:
#         stats.print_all(out=fh)
 
#     thread_stats = yappi.get_thread_stats()
#     with open('thread_stats.out', 'wb') as fh:
#         thread_stats.print_all(out=fh)

# atexit.register(finish_yappi)
# init_yappi()

# yappi stuff above here is for profiling

if __name__ == '__main__':
	main()
