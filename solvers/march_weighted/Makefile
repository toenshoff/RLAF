#CLAGS	=	 -std=c99 -O3 -static -fno-strict-aliasing -Wall
#CFLAGS	=	 -O3 -static -fno-strict-aliasing -Wall -DNDEBUG
CFLAGS	=	 -O3 -fno-strict-aliasing -Wall -DNDEBUG -fcommon
#CFLAGS	=	 -O3 -fno-strict-aliasing -Wall

#CFLAGS	= 	-Wall -O3 -ggdb -pg
#CFLAGS = 	-Wall -O3 -ggdb -pg -lefence

CC	= 	gcc-10 #gcc-4.8

.c.o:
		$(CC) -c $(CFLAGS) $(SATTEST) -o $@ $< 

march_pa: march.o \
		cube.o \
		distribution.o \
		doublelook.o \
		equivalence.o \
		memory.o \
		lookahead.o \
		parser.o \
		preselect.o \
		progressBar.o \
		resolvent.o \
		solver.o \
		transred.o \
		tree.o 
		$(CC) $(CFLAGS) $(SATTEST) *.o -lm -o march_nh
		
march.o: march.c \
		march.h \
		common.h \
		distribution.h \
		parser.h \
		solver.h \
		preselect.h \
		progressBar.h

resolvent.o: resolvent.c \
		resolvent.h \
		common.h \
		lookahead.h \
		memory.h

distribution.o: distribution.c \
		distribution.h \
		common.h

doublelook.o: doublelook.c \
		doublelook.h \
		common.h

equivalence.o: equivalence.c \
		equivalence.h \
		memory.h

memory.o: memory.c \
		memory.h \
		common.h

cube.o: cube.c \
		cube.h \
		common.h

parser.o: parser.c \
		parser.h \
		common.h \
		equivalence.h

solver.o: solver.c \
		solver.h \
		common.h \
		lookahead.h \
		preselect.h \
		progressBar.h

preselect.o: preselect.c \
		preselect.h \
		common.h

transred.o: transred.c \
		transred.h

tree.o: tree.c \
		tree.h

lookahead.o: lookahead.c \
		lookahead.h \
		doublelook.h \
		common.h \
		tree.h \
		solver.h

progressBar.o: progressBar.c \
		progressBar.h \
		common.h

clean:
	rm *.out *.py *.o -f march_nh
