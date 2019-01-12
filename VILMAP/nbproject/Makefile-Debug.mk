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
	${OBJECTDIR}/_ext/a342a8fc/ArffData.o \
	${OBJECTDIR}/_ext/c5140741/DebugOut.o \
	${OBJECTDIR}/_ext/d0624b86/Defines.o \
	${OBJECTDIR}/_ext/b4e553aa/TextToPhoneme.o \
	${OBJECTDIR}/_ext/8bffeb2f/MatUtils.o \
	${OBJECTDIR}/_ext/36bbb5bc/LHSParameters.o \
	${OBJECTDIR}/_ext/36bbb5bc/Parameters.o \
	${OBJECTDIR}/LocalLibs/Cluster/ClusteringMetrics.o \
	${OBJECTDIR}/LocalLibs/Cluster/ClusteringSOM.o \
	${OBJECTDIR}/LocalLibs/SOM/DSNeuron.o \
	${OBJECTDIR}/LocalLibs/SOM/Neuron.o \
	${OBJECTDIR}/LocalLibs/SOM/NodeW.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=-std=c++0x

# CC Compiler Flags
CCFLAGS=-Wno-comment -Wno-deprecated -std=c++11
CXXFLAGS=-Wno-comment -Wno-deprecated -std=c++11

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

${OBJECTDIR}/_ext/a342a8fc/ArffData.o: ../Libs/Data/ArffData.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/a342a8fc
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/a342a8fc/ArffData.o ../Libs/Data/ArffData.cpp

${OBJECTDIR}/_ext/c5140741/DebugOut.o: ../Libs/Debug/DebugOut.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/c5140741
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/c5140741/DebugOut.o ../Libs/Debug/DebugOut.cpp

${OBJECTDIR}/_ext/d0624b86/Defines.o: ../Libs/Defines/Defines.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/d0624b86
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/d0624b86/Defines.o ../Libs/Defines/Defines.cpp

${OBJECTDIR}/_ext/b4e553aa/TextToPhoneme.o: ../Libs/Language/TextToPhoneme.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/b4e553aa
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/b4e553aa/TextToPhoneme.o ../Libs/Language/TextToPhoneme.cpp

${OBJECTDIR}/_ext/8bffeb2f/MatUtils.o: ../Libs/MatMatrix/MatUtils.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/8bffeb2f
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/8bffeb2f/MatUtils.o ../Libs/MatMatrix/MatUtils.cpp

${OBJECTDIR}/_ext/36bbb5bc/LHSParameters.o: ../Libs/Parameters/LHSParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/36bbb5bc
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/36bbb5bc/LHSParameters.o ../Libs/Parameters/LHSParameters.cpp

${OBJECTDIR}/_ext/36bbb5bc/Parameters.o: ../Libs/Parameters/Parameters.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/36bbb5bc
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/36bbb5bc/Parameters.o ../Libs/Parameters/Parameters.cpp

${OBJECTDIR}/LocalLibs/Cluster/ClusteringMetrics.o: LocalLibs/Cluster/ClusteringMetrics.cpp
	${MKDIR} -p ${OBJECTDIR}/LocalLibs/Cluster
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LocalLibs/Cluster/ClusteringMetrics.o LocalLibs/Cluster/ClusteringMetrics.cpp

${OBJECTDIR}/LocalLibs/Cluster/ClusteringSOM.o: LocalLibs/Cluster/ClusteringSOM.cpp
	${MKDIR} -p ${OBJECTDIR}/LocalLibs/Cluster
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LocalLibs/Cluster/ClusteringSOM.o LocalLibs/Cluster/ClusteringSOM.cpp

${OBJECTDIR}/LocalLibs/SOM/DSNeuron.o: LocalLibs/SOM/DSNeuron.cpp
	${MKDIR} -p ${OBJECTDIR}/LocalLibs/SOM
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LocalLibs/SOM/DSNeuron.o LocalLibs/SOM/DSNeuron.cpp

${OBJECTDIR}/LocalLibs/SOM/Neuron.o: LocalLibs/SOM/Neuron.cpp
	${MKDIR} -p ${OBJECTDIR}/LocalLibs/SOM
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LocalLibs/SOM/Neuron.o LocalLibs/SOM/Neuron.cpp

${OBJECTDIR}/LocalLibs/SOM/NodeW.o: LocalLibs/SOM/NodeW.cpp
	${MKDIR} -p ${OBJECTDIR}/LocalLibs/SOM
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LocalLibs/SOM/NodeW.o LocalLibs/SOM/NodeW.cpp

${OBJECTDIR}/main.o: main.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../Libs/Data -I../Libs/MatMatrix -I../Libs/Language -I../Libs/Debug -I../Libs/Defines -I../Libs/Parameters -ILocalLibs/Cluster -ILocalLibs/SOM -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

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
