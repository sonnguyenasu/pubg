import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as an
#from names import get_first_name as ranName
plt.rcParams['animation.ffmpeg_path']='D:/ffmpeg-4.2.2-win64-static/ffmpeg-4.2.2-win64-static/bin/ffmpeg.exe'
np.random.seed()
img = plt.imread('pubg2.jpg')
fig, ax = plt.subplots()
n = 100
limitt = 800
#x = np.arange(-limitt,limitt,2*limitt/n)
#y = np.arange(-limitt,limitt,2*limitt/n)
kills = np.zeros(n)
labels= np.array([])
orders =np.arange(0,n,1)
names = ['FunkyM','MeoU','Aleo','Boongminz','IQ500','Jett','JJLEO','Leviz','LongK','Liningz',
'zzGou1','Tuanz','Slowz','Rambo','Nhism','Nickyyy','OcVoDich','Tik','Snake','Sapauu',
'Shroud','Jikey','BAsil','Turtle','Deftsu','Rip113','CvD','BadGuy','1405','Pika',
'ymcud','Fergus','PinoNTK','NgBaThanhDat','DJChip','Mixi','Bomman','Mimosa','TrucTiepGame','DatOC',
'Pewdeptrai','NielT','SSuBang','KenTi','Caps','Ssumday','Jindo','DucAnh','Top1Server','GIN',
'Sylphyyy','Audy','King','RedDOT69','ACE','LL','TripleL','DoesntV','Lynx','BotGiang',
'NhatThong','Sown','2Pi','Gios','Candyrox','Xemesis','TrMHung','Killernhoc7','Spring','TvT',
'HoangVu','Tab','NSN1996','Danddy','otyL','VuPK','Geosama','DoubleP','Meo2k4','Misthy',
'ThayGiaoBa','QTV','Senaaa','Kidzz','Sof13','Clear','QuangNT','Chubbyy','Viruss','tson1997',
'Taikonn','TungTT','HaiSaki','Siro','Wesneys','Fap','Dyann','Kai','Zeros','Optimus']
#labels = np.random.permutation(names)
scoress = np.zeros(n)
#try:
with open('scores.txt','r') as fr:
	for idx in range(n):
		strs = fr.readline()[:-1]
		#print(str)
		score = int(strs)
		scoress[idx] = score
#except:
#	print("Failed")

permute = np.random.permutation(range(n))
permute_inv = np.zeros(n)
scores = np.array([])
labels = np.array([])
for i in range(n):
	permute_inv[permute[i]] = i
for i in permute:
	scores = np.append(scores,scoress[i])
	labels = np.append(labels,names[i])

cop_labels = labels[:]
if np.mean(scores)==0:
	addition = np.ones(n)
else:
	addition = scores/np.mean(scores)
abilities = n*np.random.rand(n,)/10
for j in range(1,n//2+1):
	abilities[n-j] *= 1.2
for j in range(n):
	abilities[j] *= addition[j]
blood = 100*np.ones(n)
'''
x = np.array([])
y = np.array([])
for i in range(n):
	np.random.seed()
	x = np.append(x,np.random.randint(low=-limitt,high=limitt))
	y = np.append(y,np.random.randint(low=-limitt,high=limitt))
'''
x=  np.zeros(n)
y = np.zeros(n)
scat = ax.scatter(x,y,color='orange')
outside = np.array([True for i in range(n)])
survive = np.arange(0,n,1)
increment = 3
cx = None
cy = None
prev_cx= None
prev_cy= None
r = 800
prev_r=800
leng = n
top5 = []
track = []
trackgirl = []
trackboy = []
killed = [False for i in range(n)]
top = np.zeros(n)
rank = n
t = None
z1 = None
z2 = None
prev_t = None
prev_z1= None
prev_z2= None
dif = None
def animation(i):
	
	np.random.seed()
	area = {}
	global x,y,survive,outside,leng,labels,blood,kills,track,abilities,top,orders,cop_labels, killed,rank,increment,prev_r,dif
	delList = []
	if i > 60 and i % 10 ==0:
		try:
			print('Step %d %d alive : Spectating player %-15s: %.2f %.2f %d'%(i,rank,cop_labels[-1],abilities[orders[-1]],blood[-1],int(kills[orders[-1]])))
		except:
			print('No player left in the field')
	for l in range(0,len(x),1):
		if i > 60:
			try:
				a = area[(x[l]//15,y[l]//15)]
			
				ran = (abilities[orders[a]]+1+abilities[orders[l]]+1)*np.random.rand()
				if ran < abilities[orders[l]]+1:
					rem = a
				else:
					rem = l
				killer = l+a-rem
				#print(x)
				#delList.append(rem)
				blood[rem] -= max(10,abilities[orders[killer]]*2-abilities[rem]*0.7)
				abilities[rem] *= 0.9
				top[orders[rem]] = len(x)
				#print(labels[rem], "was killed by", labels[killer])
				if blood[rem] <= 0 and not killed[orders[rem]]:
					top[orders[rem]] = rank
					rank -= 1
					killed[orders[rem]] = True
					delList.append(rem)
					kills[orders[killer]] += 1
					if abilities[orders[killer]] < 50:
						abilities[orders[killer]] += 0.5*abilities[orders[rem]]
					else:
						abilities[orders[killer]] += 2
					textstr2 ='\n{} killed {}'.format(labels[orders[killer]],labels[orders[rem]])
					print(textstr2)
				
			except:
				area[(x[l]//15,y[l]//15)] = l
				#print(y[l])
	global cx,cy,r,prev_cx,prev_cy,top5,trackboy,trackgirl,t,z1,z2,prev_t,prev_z1,prev_z2
	np.random.seed(i)
	dx = 2*np.random.rand(len(x),)-1
	dy = 2*np.random.rand(len(x),)-1
	ax.clear()
	ax.imshow(img,extent=[-1.3*limitt,1.3*limitt,-1.3*limitt,1.3*limitt])
		#outside = np.array([True for j in range(len(x))])
	
	if i % 140 == 65:
		prev_cx=cx
		prev_cy=cy
		prev_r = r
		if i >= 205:
			prev_t = t[:]
			prev_z1 = z1[:]
			prev_z2 = z2[:]
		r = max(3,0.45*r)
		if not prev_cx:
			cx,cy = 2*limitt*np.random.rand(2,1)-limitt
		else:
			theta = 2*np.pi*np.random.rand()-np.pi
			ran = np.random.rand()
			dif = [(prev_r-r)*ran*np.cos(theta),(prev_r-r)*np.sin(theta)*ran]
			cx,cy = prev_cx + (prev_r-r)*np.cos(theta)*ran, prev_cy+(prev_r-r)*np.sin(theta)*ran
	if i > 65:
		t = np.arange(cx-r,cx+r+0.01,0.01)
		z1 = cy+np.sqrt(r**2-(t-cx)**2)
		z2 = cy-np.sqrt(r**2-(t-cx)**2)
		if i <= 205 or i %140 >= 110 or i%140 < 65:
			ax.plot(t,z1,c='b')
			ax.plot(t,z2,c='b')
		elif i > 205 and i %140 >= 65:
			prev_prev_cx = prev_cx
			prev_prev_cy = prev_cy
			prev_prev_r = prev_r
			prev_cx,prev_cy = prev_cx+dif[0]/45,prev_cy+dif[1]/45
			prev_r = r/0.45- r*0.55*((i%140-65)%45)/(45*0.45)
			prev_t = np.arange(prev_cx-prev_r, prev_cx+prev_r+0.01,0.01)
			prev_z1 = prev_cy+np.sqrt(prev_r**2-(prev_t-prev_cx)**2)
			prev_z2 = prev_cy-np.sqrt(prev_r**2-(prev_t-prev_cx)**2)
			ax.plot(prev_t,prev_z1,c='b')
			ax.plot(prev_t,prev_z2,c='b')
			ax.plot(t,z1,c='white')
			ax.plot(t,z2,c='white')
		for j in range(len(x)):
			if (x[j]-prev_cx)**2 + (y[j]-prev_cy)**2 < prev_r**2:
				outside[j] = False
			else:
				outside[j] = True
	if i < 75:
		ran = np.random.randint(5,60,size=(len(x),))
		x += np.multiply(ran,dx>0)-np.multiply(ran,dx<=0)
		y += np.multiply(ran,dy>0)-np.multiply(ran,dy<=0)
	
	elif i >= 75:
		choice = np.random.randint(55,size=(len(x),))
		inc = np.multiply((choice <= (orders%50)/2),(1-outside))
		x -= np.multiply(np.sign(x-cx)*30000/(i*np.sqrt(np.dot(x+y-cx-cy,x+y-cx-cy)))*increment,inc)
		y -= np.multiply(np.sign(y-cy)*30000/(i*np.sqrt(np.dot(x+y-cx-cy,x+y-cx-cy)))*increment,inc)
		if i >= 100:
			abilities[orders] += 0.01*inc
		else:
			abilities[orders] += 0.03*inc
	#for j in range(len(x)):
		#if i < 100 or not outside[j]:
			#pass
			
		#elif dx[j] > 0 and not cx:
		x += (not cx)*increment*(dx>0)-(not cx)*increment*(dx<=0)-(not not cx)*np.multiply(np.sign(x-cx)*increment*30000/np.sqrt(i/2*np.dot(x+y-cx-cy,x+y-cx-cy)),outside)
		y += (not cx)*increment*(dy>0)-(not cx)*increment*(dy<=0)-(not not cx)*np.multiply(np.sign(y-cy)*30000/np.sqrt(i/2*np.dot(x+y-cx-cy,x+y-cx-cy))*increment,outside)
			
		#elif dx[j] <= 0 and not cx:
			#x[j] -= increment
		#else:
			#x[j] -= np.sign(x[j]-cx)*increment
		#if not outside[j]:
			#pass
		#elif dy[j] > 0 and not cx:
		#	y[j] += increment
		#elif dy[j] <= 0 and not cx:
		#	y[j] -= increment
		#else:
		#	y[j] -= np.sign(y[j]-cy)*increment
	
	for j in range(len(x)):
		if killed[orders[j]]:
			if (j not in delList):
				top[orders[j]] = rank
				rank -= 1
				delList.append(j)
		elif blood[j] <= 0:
			textstr2=cop_labels[j]+" died outside the playzone"
			print(textstr2)
			delList.append(j)
			killed[j] = True
			top[orders[j]] = rank
			rank -= 1
		elif outside[j] and i%140>110:
			blood[j] -= 400/r
		else:
			blood[j] += increment*min(60/r,r/3000)*((orders[j] >= 50)+1)
			blood[j] = min(100,blood[j])
	
	x = np.delete(x,delList)
	y = np.delete(y,delList)
	blood = np.delete(blood,delList)
	orders = np.delete(orders, delList)
	cop_labels = np.delete(cop_labels,delList)
	survive= np.delete(survive,delList)
	outside = np.delete(outside,delList)
	textstr = '\n{} ALIVE'.format(len(x))
	if not prev_cx:
		prev_cx = 0
		prev_cy = 0
	try:
		ax.scatter(x[:-1],y[:-1],c='orange')
		ax.scatter(x[-1],y[-1],c='red')
	except:
		ax.scatter(x,y,c='orange')
	ax.text(0.05,0.95,textstr,transform=ax.transAxes,fontsize=10,c='white')
	try:
		ax.set_title(textstr2)
	except:
		ax.set_title(i)
	if r > 600:
		ax.set_xlim(-1.1*limitt,1.1*limitt)
		ax.set_ylim(-1.1*limitt,1.1*limitt)
	elif r <= 600:
		ax.set_xlim(max(-1.2*limitt,prev_cx-5*r),min(1.2*limitt,prev_cx+5*r))
		ax.set_ylim(max(-1.2*limitt,prev_cy-5*r),min(1.2*limitt,prev_cy+5*r))
	ax.scatter(prev_cx,prev_cy,marker='x')
	#ax.set_title("i: "+str(i))
	#split = min(n//2,len(x)-1)
	#data = np.hstack((x[:,np.newaxis],y[:,np.newaxis]))
	#scat.set_offsets(data)
	for k in range(len(x)):
		ax.annotate(cop_labels[k],(x[k],y[k]),c='white')	
	#ax.show()
	
	if len(survive) == 5:
		top5 = survive[:]
	track.append(rank)	
	boy = [j for j in orders if j<n/2]
	trackboy.append(len(boy))
	trackgirl.append(len(x)-len(boy))
	return ax,
anim = FuncAnimation(fig, animation, frames=1000, interval=100,repeat=False) 
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
#Writer = an.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#anim.save('game1.mp4',writer=writer)
#plt.show()
if len(survive) == 1:
	print("WINNER WINNER CHICKEN DINNER")
	print(labels[survive[0]], "was the winner for this round with",kills[survive[0]],'kills')
	top[survive[0]] = 1
	mostKills = np.argmax(kills)
	print("Top KILLER:",labels[mostKills],'with',kills[mostKills],'KILLS')
	#print("TOP 5:")
for i in range(n):
	scores[i] += (300-top[i]*3)+kills[i]
#j = np.argmax(scores)
#print("Best player of this round:",labels[j])
top5_scores = np.argsort(-scores)[:5]
a = 'username'
print("TOP %10s Scores"%a)
for idx,i in enumerate(top5_scores):
	print("%3d %10s %4d points"%(idx+1,labels[i],scores[i]))
plt.plot(track)
plt.plot(trackgirl,label='safe')
plt.plot(trackboy,label='aggressive')
plt.legend(loc='upper right')
plt.show()

top_inv = np.zeros(n)

for idx, val in enumerate(top):
	top_inv[int(val)-1] = idx

for i in range(n):
	print("Player %-15s: %-4d %3d kills"%(labels[int(top_inv[i])],i+1,int(kills[int(top_inv[i])])))

with open('scores.txt','w') as f:
	for i in range(n):
		f.write(str(int(scores[int(permute_inv[i])]))+'\n')

'''
circle: (z-cx)**2 + (t-cy)**2 = r**2
z1 = cy+np.sqrt(r**2-(t-cx)**2)
z2 = cy-np.sqrt(r**2-(t-cx)**2)

'''