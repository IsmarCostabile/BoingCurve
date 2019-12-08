# load packages
from matplotlib import pyplot as plt
import numpy as np
from sympy import *
from matplotlib.animation import FuncAnimation

# Ready
print("Ready")

# define figure and axes and set some graphing parameters
fig, ax = plt.subplots(figsize = (4, 4))
ax.set_aspect(1)
plt.rc('font', size = 16)
marker_style = dict(linestyle = 'none', markersize = 10, markeredgecolor = 'k')
plt.box(on = None)
fig.set_facecolor('#c0c0c0')
plt.rcParams['savefig.facecolor'] = '#c0c0c0'
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')

# plot the axes
(min_plot, max_plot) = (0, 10)
plt.xticks(np.arange(min_plot, max_plot + 1, 2), fontsize = 10)
plt.yticks(np.arange(min_plot, max_plot + 1, 2), fontsize = 10)
plt.xlim([min_plot - 1, max_plot + 1])
plt.ylim([min_plot - 1, max_plot + 1])
ax.annotate('', xy = (min_plot, max_plot + 1), xytext = (min_plot, min_plot - 0.1), 
	arrowprops = {'arrowstyle': '->', 'lw': 1, 'facecolor': 'k'}, va = 'center')
ax.annotate('', xy = (max_plot + 1, min_plot), xytext = (min_plot - 0.1, min_plot), 
	arrowprops = {'arrowstyle': '->', 'lw': 1, 'facecolor': 'k'}, va = 'center', zorder = -1) 
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.0, hspace=0.0)

# randomly generate some data and add an underlying function to make it interesting
no_of_points = 15
(mean_x, sd_x) = (5, 5)
(mean_y, sd_y) = (6, 0.70)
data_x = np.random.normal(mean_x, sd_x, no_of_points)
data_y = np.random.normal(mean_y, sd_y, no_of_points)
for idx, val in enumerate(data_y): # add an arbitrary cubic function
	data_y[idx] = (data_y[idx] + (data_x[idx] - (max_plot - min_plot) / 2) / 2
		- 1 / 10 * (data_x[idx] - (max_plot - min_plot) / 2)**3)
# keep only data points that lie inside the plotted area
check_limits_array = np.transpose(np.array([data_x, data_y]))
data_x = [x[0] for x in check_limits_array if 
	min_plot < x[0] < max_plot and min_plot < x[1] < max_plot]
data_y = [x[1] for x in check_limits_array if 
	min_plot < x[0] < max_plot and min_plot < x[1] < max_plot]
(average_x, average_y) = (np.average(data_x), np.average(data_y))
no_of_points = len(data_x)

# fit the slope and intercept of the line by least squares
# then fit a polynomial and get the x spacing to plot it

data_slope, data_intercept = np.polyfit(data_x, data_y, 1)
if data_slope == 0: # fix reciprocal problem if the slope is 0
	data_slope = 1e-6
polynomial_degree = 3
data_poly_fit = np.polyfit(data_x, data_y, polynomial_degree)
data_poly_eval = np.poly1d(data_poly_fit)
plot_point_number = 50
x_spaced = np.linspace(min_plot, max_plot, plot_point_number)

# pick some initial fitting line properties; 
# the line will pop up from this arrangement

line_intercept_0 = 1
line_slope_0 = 0
if line_slope_0 == 0: # fix reciprocal problem if the slope is 0
	line_slope_0 = 1e-6

# write a function to get the distance from any point to the fitting line;
# also get the difference in slope between the line and the best fit to the data
# and the moving fitting line and return the point 
# on the fitting line where the perpendicular line should terminate 

def get_line_distance_and_perp(line_slope, line_intercept, data_point):
	# transform line equation into Hess form
	(w1, w2, b) = (-line_slope, 1, -line_intercept)
	# apply the distance formula d = (wÂ·x+b)/||w|| to get the distance
	# between where we are and where we want to be
	line_distance_diff = (np.array((w1, w2)) @ np.array(data_point) + b) / np.linalg.norm((w1, w2))
	# also get the difference in slope between where we are 
	# and where we want to be
	slope_angle_diff = np.arctan2(line_slope, 1)-np.arctan2(data_slope, 1)
	line_perp_slope = -1/line_slope
	line_perp_angle = np.arctan2(line_perp_slope, 1)
	# fix a problem with the connecting lines sometimes pointing in the wrong direction
	line_distance_diff = line_distance_diff * line_slope / abs(line_slope)
	(line_point_x, line_point_y) = (data_point[0] + line_distance_diff * np.cos(line_perp_angle), 
		data_point[1] + line_distance_diff * np.sin(line_perp_angle))
	return line_distance_diff, slope_angle_diff, (line_point_x, line_point_y)

# run that function to get the initial distance and slope difference information

(line_distance_diff, slope_angle_diff) = get_line_distance_and_perp(line_slope_0, 
		line_intercept_0, (average_x, average_y))[0:2]
	
# write an oscillation function that overshoots and then settles down

def get_osc_funct(time): # set some spring, mass, damper values that provide pleasing oscillation
	k = 4 # spring value
	m = 1 # mass value
	c = 1.2 # damper value
	wn = np.sqrt(k / m) # natural frequency
	z = c / (2 * m * wn) # damping coefficient
	wd = np.sqrt(1 - z**2) * wn # damped natural frequency
	# two equivalent functions to find the displacement over time; either works
	osc_funct1 = np.exp(-z * wn * time) * (np.cos(wd * time) + z / np.sqrt(1 - z**2) * np.sin(wd * time))
	# osc_funct2 = np.exp(-z * wn * time) * np.sqrt(1 / (1 - z**2)) * np.cos(wd * time - np.arctan(z / np.sqrt(1 - z**2)))
	return osc_funct1

def get_line(line_distance_diff, slope_angle_diff, data_point, time, parameter):
	# we can apply the pleasing oscillation to either the distance or the slope
	if parameter == "int": # here, the distance oscillates; 
	# return the appropriate fitting line slope and intercept
		line_distance = line_distance_diff * get_osc_funct(time)
		line_slope = line_slope_0
		line_perp_slope = -1 / line_slope_0
		line_perp_angle = np.arctan(line_perp_slope)
		(line_point_x, line_point_y) = (data_point[0] + line_distance * np.cos(line_perp_angle), 
			data_point[1] + line_distance * np.sin(line_perp_angle))
		line_intercept = line_point_y - line_slope_0 * line_point_x
	elif parameter == "slope": # here, the angle oscillates; again, 
	# return the appropriate fitting line slope and intercept
		slope_angle_difference = slope_angle_diff * get_osc_funct(time)
		line_slope = np.tan(slope_angle_difference + np.arctan(data_slope))
		line_intercept = data_point[1] - line_slope * data_point[0]
	return line_slope, line_intercept

# get the vertical distances between the curve and the straight-line fit;
# these vertical distances will oscillate, and a polynomial will be repeatedly fit to them to 
# produce the transition from the fitted line to the fitted polynomial

polynomial_spacing = polynomial_degree + 2 # the number of points used to make the polynomial oscillation smooth
poly_x_spaced = np.linspace(min_plot, max_plot, polynomial_spacing)
poly_distances = [data_poly_eval(a) - (a * data_slope + data_intercept) for a in poly_x_spaced]

# write a function to get the best-fitting polynomial as we oscillate around the final polynomial

def get_polynomial(data_slope, data_intercept, poly_distances, time):
	trans_poly_distances = [x * (1 - get_osc_funct(time)) for x in poly_distances]
	trans_poly_x = [a for a, b in zip(poly_x_spaced, trans_poly_distances)]
	trans_poly_y = [data_slope * a + data_intercept + b for a, b in zip(poly_x_spaced, trans_poly_distances)]
	trans_poly_fit = np.polyfit(trans_poly_x, trans_poly_y, polynomial_degree)
	trans_poly_eval = np.poly1d(trans_poly_fit)
	return trans_poly_fit, trans_poly_eval

# write a function to get the closest point between a data point and a polynomial

def get_closest_point(poly_fit, poly_eval, data_point):
	xx = symbols("xx")
	list0 = np.ndarray.tolist(poly_fit)
	poly_fit_sym = Poly(list0, xx)
	# take the derivative of the fitted polynomial
	poly_deriv_sym = poly_fit_sym.diff(xx)
	# set the following term to zero to minimize the (square of the) distance between the point and polynomial
	equation = xx - data_point[0] + (poly_fit_sym - data_point[1]) * poly_deriv_sym
	coeff = Poly(equation).all_coeffs()
	solved = np.roots(coeff)
	# chop any tiny imaginary component arising from the numerical solution
	solved = solved.real[abs(solved.imag) < 1e-6]
	dist = [(n - data_point[0])**2 + (data_poly_eval(n) - data_point[1])**2 for n in solved] # get distances
	solved = solved[np.argmin(dist)] # get shortest distance if there's more than one
	return solved, poly_eval(solved)

# now make the fitting lines - the first one moves; 
# the second fades back in at the end

fit_line_color = '0.5'
component_1_line, = ax.plot((min_plot, max_plot), 
	(min_plot * line_slope_0 + line_intercept_0, max_plot * line_slope_0 + line_intercept_0), 
	lw = 3, color = fit_line_color, zorder = 0)
component_2_line, = ax.plot((min_plot, max_plot), 
	(min_plot * line_slope_0 + line_intercept_0, max_plot * line_slope_0 + line_intercept_0), 
	lw = 3, color = fit_line_color, zorder = -2)

# and make thin perpendicular connector lines 
# between each point and the fitting line

connector_line = []
for i in range(len(data_x)):
	(line_point_x, line_point_y) = get_line_distance_and_perp(line_slope_0, 
		line_intercept_0, (data_x[i], data_y[i]))[2]
	connector_line_component, = ax.plot((data_x[i], line_point_x), 
		(data_y[i], line_point_y), 
		lw = 1, color = 'r', zorder = -1)
	connector_line.append(connector_line_component)

# plot the data points

ax.plot(data_x, data_y, marker = "o", 
	markerfacecolor = 'red', **marker_style, zorder = 0)

# set the animation parameters, line states, and expected motion paths

smoothness = 1 # controls the temporal resolution; essentially a multiple of
# the frame rate; 1 is good for prototyping, and 5 is very smooth

# define the duration of each segment of the fitting animation

frame_list = [['fadein', 6],
	['fadein_pause', 2],
	['intercept_oscillate', 8],
	['intercept_pause', 0],
	['slope_oscillate', 8],
	['slope_pause', 2],
	['poly_oscillate', 8],
	['poly_pause', 2],
	['fadeout', 6],
	['fadeout_pause', 2]]

# define some useful frame parameters

frame_names = [i[0] for i in frame_list]
frame_durations = [smoothness * i[1] for i in frame_list]

# define some useful time references

def time_past(period): # total time before a certain segment
	return sum(frame_durations[0:frame_names.index(period)])
def time_future(period): # total time including a certain segment
	return sum(frame_durations[0:frame_names.index(period) + 1])
def time_present(period): # duration of a certain segment
	return frame_durations[frame_names.index(period)]
def norm_time(i, period): # how far we are into a certain segment
	return (i - time_past(period)) / time_present(period)
def effective_time(i, period): # used as the time in the oscillation function
	return (i - time_past(period)) / smoothness
		
# define the aspects of the animation function

def init():
	return (component_1_line, component_2_line, connector_line,)

print("Starting to accumulate animation frames")

def update(i):
	print(f"Frame: {i+1}")
	if i == 0:
		component_2_line.set_alpha(0) # the second line is set as invisible
		component_1_line.set_alpha(1) # the first line is set as visible
		for j in range(len(data_x)): # set all the connector lines
			(line_point_x, line_point_y) = get_line_distance_and_perp(line_slope_0, 
				line_intercept_0, (data_x[j], data_y[j]))[-1]
			connector_line[j].set_xdata((data_x[j], line_point_x))
			connector_line[j].set_ydata((data_y[j], line_point_y))
	if i <= time_future('fadein'): # this is the component line fade-in part
		for j in range(len(data_x)): # the connector lines fade in
			connector_line[j].set_alpha(min(norm_time(i, 'fadein'), 1))
	elif i <= time_future('fadein_pause'): # pause if desired
		pass
	elif i <= time_future('intercept_oscillate'): # this the line translation part
		(line_slope, line_intercept) = get_line(line_distance_diff, slope_angle_diff, 
			(average_x, average_y), effective_time(i, 'intercept_oscillate'), "int")[0:2]
		component_1_line.set_ydata((min_plot * line_slope + line_intercept, max_plot * line_slope + line_intercept))
		for j in range(len(data_x)):
			(line_point_x, line_point_y) = get_line_distance_and_perp(line_slope, 
				line_intercept, (data_x[j], data_y[j]))[-1]
			connector_line[j].set_xdata((data_x[j], line_point_x))
			connector_line[j].set_ydata((data_y[j], line_point_y))
	elif i <= time_future('intercept_pause'): # pause if desired
		pass
	elif i <= time_future('slope_oscillate'): # this is the line rotation part
		(line_slope, line_intercept) = get_line(0, slope_angle_diff, (average_x, average_y), 
			effective_time(i, 'slope_oscillate'), "slope")[0:2]
		component_1_line.set_ydata((min_plot * line_slope + line_intercept, max_plot * line_slope + line_intercept))
		for j in range(len(data_x)):
			(line_point_x, line_point_y) = get_line_distance_and_perp(line_slope, 
				line_intercept, (data_x[j], data_y[j]))[-1]
			connector_line[j].set_xdata((data_x[j], line_point_x))
			connector_line[j].set_ydata((data_y[j], line_point_y))
	elif i <= time_future('slope_pause'):	# pause if desired
		pass
	elif i <= time_future('poly_oscillate'): # this is the polynomial part
		component_1_line.set_xdata(x_spaced)
		(trans_poly_fit, trans_poly_eval) = get_polynomial(data_slope, data_intercept, poly_distances, 
			effective_time(i, 'poly_oscillate'))
		component_1_line.set_ydata(trans_poly_eval(x_spaced))
		for j in range(len(data_x)):
	 		(x_closest_point, y_closest_point) = get_closest_point(trans_poly_fit, trans_poly_eval, (data_x[j], data_y[j]))
	 		connector_line[j].set_xdata((data_x[j], x_closest_point))
	 		connector_line[j].set_ydata((data_y[j], y_closest_point))
	elif i <= time_future('poly_pause'): # pause if desired
		pass
	elif i <= time_future('fadeout'): 
		# fade from the correctly fitted line to the original line
		# speed up the fade-out a little relative to the fade-in
		fade_speedup = 2
		for j in range(len(data_x)):
			connector_line[j].set_alpha(max(1 -	fade_speedup * norm_time(i, 'fadeout'), 0))
		component_1_line.set_alpha(max(1 - fade_speedup * norm_time(i, 'fadeout'), 0))
		component_2_line.set_alpha(min(norm_time(i, 'fadeout'), 1))
	elif i <= time_future('fadeout_pause'): # pause if desired 
		pass		
	return (component_1_line, component_2_line, connector_line,)

# implement the animation and save the results 

anim = FuncAnimation(fig, update, frames = sum(frame_durations), interval = 100 / smoothness, init_func = init, blit = False)
anim.save('animation.gif', dpi = 72, writer = 'imagemagick')

print('Done')
