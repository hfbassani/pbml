#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/_ext/a845ff68/Cluster.o \
	${OBJECTDIR}/_ext/a845ff68/ClusteringMetrics.o \
	${OBJECTDIR}/_ext/a845ff68/ClusteringSOM.o \
	${OBJECTDIR}/_ext/a845ff68/SSCDataFile.o \
	${OBJECTDIR}/_ext/a845ff68/randomnumbers.o \
	${OBJECTDIR}/_ext/a342a8fc/ArffData.o \
	${OBJECTDIR}/_ext/c5140741/DebugOut.o \
	${OBJECTDIR}/_ext/d0624b86/Defines.o \
	${OBJECTDIR}/_ext/b4e553aa/TextToPhoneme.o \
	${OBJECTDIR}/_ext/8bffeb2f/MatUtils.o \
	${OBJECTDIR}/_ext/36bbb5bc/LHSParameters.o \
	${OBJECTDIR}/_ext/36bbb5bc/Parameters.o \
	${OBJECTDIR}/_ext/15c88dff/DSNeuron.o \
	${OBJECTDIR}/_ext/15c88dff/DSSOM.o \
	${OBJECTDIR}/_ext/15c88dff/Neuron.o \
	${OBJECTDIR}/_ext/15c88dff/NodeW.o \
	${OBJECTDIR}/_ext/15c88dff/SOM2D.o \
	${OBJECTDIR}/_ext/15c88dff/SOMAW.o \
	${OBJECTDIR}/_ext/15c88dff/TDSSOM.o \
	${OBJECTDIR}/_ext/15c88dff/TSOM.o \
	${OBJECTDIR}/MyParameters/MyParameters.o \
	${OBJECTDIR}/OutputMetrics/OutputMetrics.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=-std=c++0x

# CC Compiler Flags
CCFLAGS=-std=c++11
CXXFLAGS=-std=c++11

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/vilmap

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/vilmap: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/vilmap ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/_ext/a845ff68/Cluster.o: ../Libs/Cluster/Cluster.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/a845ff68
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/a845ff68/Cluster.o ../Libs/Cluster/Cluster.cpp

${OBJECTDIR}/_ext/a845ff68/ClusteringMetrics.o: ../Libs/Cluster/ClusteringMetrics.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/a845ff68
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/a845ff68/ClusteringMetrics.o ../Libs/Cluster/ClusteringMetrics.cpp

${OBJECTDIR}/_ext/a845ff68/ClusteringSOM.o: ../Libs/Cluster/ClusteringSOM.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/a845ff68
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/a845ff68/ClusteringSOM.o ../Libs/Cluster/ClusteringSOM.cpp

${OBJECTDIR}/_ext/a845ff68/SSCDataFile.o: ../Libs/Cluster/SSCDataFile.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/a845ff68
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/a845ff68/SSCDataFile.o ../Libs/Cluster/SSCDataFile.cpp

${OBJECTDIR}/_ext/a845ff68/randomnumbers.o: ../Libs/Cluster/randomnumbers.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/a845ff68
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/a845ff68/randomnumbers.o ../Libs/Cluster/randomnumbers.cpp

${OBJECTDIR}/_ext/a342a8fc/ArffData.o: ../Libs/Data/ArffData.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/a342a8fc
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/a342a8fc/ArffData.o ../Libs/Data/ArffData.cpp

${OBJECTDIR}/_ext/c5140741/DebugOut.o: ../Libs/Debug/DebugOut.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/c5140741
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/c5140741/DebugOut.o ../Libs/Debug/DebugOut.cpp

${OBJECTDIR}/_ext/d0624b86/Defines.o: ../Libs/Defines/Defines.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/d0624b86
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/d0624b86/Defines.o ../Libs/Defines/Defines.cpp

${OBJECTDIR}/_ext/b4e553aa/TextToPhoneme.o: ../Libs/Language/TextToPhoneme.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/b4e553aa
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/b4e553aa/TextToPhoneme.o ../Libs/Language/TextToPhoneme.cpp

${OBJECTDIR}/_ext/8bffeb2f/MatUtils.o: ../Libs/MatMatrix/MatUtils.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/8bffeb2f
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/8bffeb2f/MatUtils.o ../Libs/MatMatrix/MatUtils.cpp

${OBJECTDIR}/_ext/36bbb5bc/LHSParameters.o: ../Libs/Parameters/LHSParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/36bbb5bc
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/36bbb5bc/LHSParameters.o ../Libs/Parameters/LHSParameters.cpp

${OBJECTDIR}/_ext/36bbb5bc/Parameters.o: ../Libs/Parameters/Parameters.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/36bbb5bc
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/36bbb5bc/Parameters.o ../Libs/Parameters/Parameters.cpp

${OBJECTDIR}/_ext/15c88dff/DSNeuron.o: ../Libs/SOM/DSNeuron.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/15c88dff
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/15c88dff/DSNeuron.o ../Libs/SOM/DSNeuron.cpp

${OBJECTDIR}/_ext/15c88dff/DSSOM.o: ../Libs/SOM/DSSOM.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/15c88dff
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/15c88dff/DSSOM.o ../Libs/SOM/DSSOM.cpp

${OBJECTDIR}/_ext/15c88dff/Neuron.o: ../Libs/SOM/Neuron.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/15c88dff
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/15c88dff/Neuron.o ../Libs/SOM/Neuron.cpp

${OBJECTDIR}/_ext/15c88dff/NodeW.o: ../Libs/SOM/NodeW.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/15c88dff
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/15c88dff/NodeW.o ../Libs/SOM/NodeW.cpp

${OBJECTDIR}/_ext/15c88dff/SOM2D.o: ../Libs/SOM/SOM2D.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/15c88dff
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/15c88dff/SOM2D.o ../Libs/SOM/SOM2D.cpp

${OBJECTDIR}/_ext/15c88dff/SOMAW.o: ../Libs/SOM/SOMAW.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/15c88dff
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/15c88dff/SOMAW.o ../Libs/SOM/SOMAW.cpp

${OBJECTDIR}/_ext/15c88dff/TDSSOM.o: ../Libs/SOM/TDSSOM.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/15c88dff
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/15c88dff/TDSSOM.o ../Libs/SOM/TDSSOM.cpp

${OBJECTDIR}/_ext/15c88dff/TSOM.o: ../Libs/SOM/TSOM.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/15c88dff
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/15c88dff/TSOM.o ../Libs/SOM/TSOM.cpp

${OBJECTDIR}/MyParameters/MyParameters.o: MyParameters/MyParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/MyParameters
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MyParameters/MyParameters.o MyParameters/MyParameters.cpp

${OBJECTDIR}/OutputMetrics/OutputMetrics.o: OutputMetrics/OutputMetrics.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputMetrics
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/OutputMetrics/OutputMetrics.o OutputMetrics/OutputMetrics.cpp

${OBJECTDIR}/main.o: main.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/MatMatrix -I../Libs/SOM -I../Libs/Parameters -I../Libs/Debug -I../Libs/Defines -I../Libs/CImg -I../Libs/Cluster -I../Libs/Data -I../Libs/Language -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
