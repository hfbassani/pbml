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
	${OBJECTDIR}/_ext/28982ceb/ArffData.o \
	${OBJECTDIR}/_ext/ea6f0132/DebugOut.o \
	${OBJECTDIR}/_ext/ae68d37/Defines.o \
	${OBJECTDIR}/_ext/ae68d37/StringHelper.o \
	${OBJECTDIR}/_ext/367a84a0/MatUtils.o \
	${OBJECTDIR}/_ext/db944a6b/LHSParameters.o \
	${OBJECTDIR}/_ext/db944a6b/LatinHypercubeSampling.o \
	${OBJECTDIR}/_ext/db944a6b/Parameters.o \
	${OBJECTDIR}/_ext/33092c8b/Cluster.o \
	${OBJECTDIR}/_ext/33092c8b/ClusteringMetrics.o \
	${OBJECTDIR}/_ext/33092c8b/ClusteringSOM.o \
	${OBJECTDIR}/_ext/33092c8b/SSCDataFile.o \
	${OBJECTDIR}/_ext/33092c8b/SubspaceClusteringSOM.o \
	${OBJECTDIR}/_ext/33092c8b/randomnumbers.o \
	${OBJECTDIR}/_ext/511dc4a2/DSNode.o \
	${OBJECTDIR}/_ext/511dc4a2/NodeW.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/netbeansproject

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/netbeansproject: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/netbeansproject ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/_ext/28982ceb/ArffData.o: ../../Libs/Data/ArffData.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/28982ceb
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/28982ceb/ArffData.o ../../Libs/Data/ArffData.cpp

${OBJECTDIR}/_ext/ea6f0132/DebugOut.o: ../../Libs/Debug/DebugOut.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/ea6f0132
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/ea6f0132/DebugOut.o ../../Libs/Debug/DebugOut.cpp

${OBJECTDIR}/_ext/ae68d37/Defines.o: ../../Libs/Defines/Defines.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/ae68d37
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/ae68d37/Defines.o ../../Libs/Defines/Defines.cpp

${OBJECTDIR}/_ext/ae68d37/StringHelper.o: ../../Libs/Defines/StringHelper.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/ae68d37
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/ae68d37/StringHelper.o ../../Libs/Defines/StringHelper.cpp

${OBJECTDIR}/_ext/367a84a0/MatUtils.o: ../../Libs/MatMatrix/MatUtils.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/367a84a0
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/367a84a0/MatUtils.o ../../Libs/MatMatrix/MatUtils.cpp

${OBJECTDIR}/_ext/db944a6b/LHSParameters.o: ../../Libs/Parameters/LHSParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/db944a6b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/db944a6b/LHSParameters.o ../../Libs/Parameters/LHSParameters.cpp

${OBJECTDIR}/_ext/db944a6b/LatinHypercubeSampling.o: ../../Libs/Parameters/LatinHypercubeSampling.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/db944a6b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/db944a6b/LatinHypercubeSampling.o ../../Libs/Parameters/LatinHypercubeSampling.cpp

${OBJECTDIR}/_ext/db944a6b/Parameters.o: ../../Libs/Parameters/Parameters.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/db944a6b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/db944a6b/Parameters.o ../../Libs/Parameters/Parameters.cpp

${OBJECTDIR}/_ext/33092c8b/Cluster.o: ../Cluster/Cluster.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/33092c8b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/33092c8b/Cluster.o ../Cluster/Cluster.cpp

${OBJECTDIR}/_ext/33092c8b/ClusteringMetrics.o: ../Cluster/ClusteringMetrics.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/33092c8b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/33092c8b/ClusteringMetrics.o ../Cluster/ClusteringMetrics.cpp

${OBJECTDIR}/_ext/33092c8b/ClusteringSOM.o: ../Cluster/ClusteringSOM.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/33092c8b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/33092c8b/ClusteringSOM.o ../Cluster/ClusteringSOM.cpp

${OBJECTDIR}/_ext/33092c8b/SSCDataFile.o: ../Cluster/SSCDataFile.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/33092c8b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/33092c8b/SSCDataFile.o ../Cluster/SSCDataFile.cpp

${OBJECTDIR}/_ext/33092c8b/SubspaceClusteringSOM.o: ../Cluster/SubspaceClusteringSOM.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/33092c8b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/33092c8b/SubspaceClusteringSOM.o ../Cluster/SubspaceClusteringSOM.cpp

${OBJECTDIR}/_ext/33092c8b/randomnumbers.o: ../Cluster/randomnumbers.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/33092c8b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/33092c8b/randomnumbers.o ../Cluster/randomnumbers.cpp

${OBJECTDIR}/_ext/511dc4a2/DSNode.o: ../SOM/DSNode.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/511dc4a2
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/511dc4a2/DSNode.o ../SOM/DSNode.cpp

${OBJECTDIR}/_ext/511dc4a2/NodeW.o: ../SOM/NodeW.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/511dc4a2
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/511dc4a2/NodeW.o ../SOM/NodeW.cpp

${OBJECTDIR}/main.o: main.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/MatMatrix -I../../Libs/Defines -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Parameters -I../SOM -I../Cluster -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

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
