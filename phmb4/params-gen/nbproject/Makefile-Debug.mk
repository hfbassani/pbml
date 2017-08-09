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
	${OBJECTDIR}/_ext/db944a6b/LHSParameters.o \
	${OBJECTDIR}/_ext/db944a6b/LatinHypercubeSampling.o \
	${OBJECTDIR}/_ext/db944a6b/Parameters.o \
	${OBJECTDIR}/MyParameters/MyParameters.o \
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
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/params-gen

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/params-gen: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/params-gen ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/_ext/db944a6b/LHSParameters.o: ../../Libs/Parameters/LHSParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/db944a6b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/CImg -I../../Libs/Cluster -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Defines -I../../Libs/Language -I../../Libs/Parameters -I../../Libs/SOM -I../../Libs/MatMatrix -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/db944a6b/LHSParameters.o ../../Libs/Parameters/LHSParameters.cpp

${OBJECTDIR}/_ext/db944a6b/LatinHypercubeSampling.o: ../../Libs/Parameters/LatinHypercubeSampling.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/db944a6b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/CImg -I../../Libs/Cluster -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Defines -I../../Libs/Language -I../../Libs/Parameters -I../../Libs/SOM -I../../Libs/MatMatrix -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/db944a6b/LatinHypercubeSampling.o ../../Libs/Parameters/LatinHypercubeSampling.cpp

${OBJECTDIR}/_ext/db944a6b/Parameters.o: ../../Libs/Parameters/Parameters.cpp
	${MKDIR} -p ${OBJECTDIR}/_ext/db944a6b
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/CImg -I../../Libs/Cluster -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Defines -I../../Libs/Language -I../../Libs/Parameters -I../../Libs/SOM -I../../Libs/MatMatrix -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/db944a6b/Parameters.o ../../Libs/Parameters/Parameters.cpp

${OBJECTDIR}/MyParameters/MyParameters.o: MyParameters/MyParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/MyParameters
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/CImg -I../../Libs/Cluster -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Defines -I../../Libs/Language -I../../Libs/Parameters -I../../Libs/SOM -I../../Libs/MatMatrix -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MyParameters/MyParameters.o MyParameters/MyParameters.cpp

${OBJECTDIR}/main.o: main.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../Libs/CImg -I../../Libs/Cluster -I../../Libs/Data -I../../Libs/Debug -I../../Libs/Defines -I../../Libs/Language -I../../Libs/Parameters -I../../Libs/SOM -I../../Libs/MatMatrix -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

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
