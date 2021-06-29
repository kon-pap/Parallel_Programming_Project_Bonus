CXX=c++
CXX_FLAGS= -O3 -std=c++17 -lm -Wall -Wextra -mavx
OPENMP = -fopenmp 

MPICXX = mpicxx
MPICXX_FLAGS = --std=c++17 -mavx -O3 -Wall -Wextra -g -DOMPI_SKIP_MPICXX
# this compiler definition is needed to silence warnings caused by the openmpi CXX
# bindings that are deprecated. This is needed on gnu compilers from version 8 forward.
# see: https://github.com/open-mpi/ompi/issues/5157

all: sequential omp mpi hybrid

#-----------------------------------------------------------------------------------------#
sequential: kdtree_sequential.cpp Node.cpp Node.hpp
	$(CXX) $(CXX_FLAGS) -o sequential kdtree_sequential.cpp Node.cpp Utility.cpp

run_sequential:
	./sequential

gprof_sequential: kdtree_sequential.cpp Node.cpp Node.hpp
	$(CXX) $(CXX_FLAGS) -pg -o sequential kdtree_sequential.cpp Node.cpp Utility.cpp
	echo "1" | ./sequential 
	gprof sequential gmon.out > gprof_sequential_analysis.prof

perf_sequential: kdtree_sequential.cpp Node.cpp Node.hpp
	$(CXX) $(CXX_FLAGS) -o sequential kdtree_sequential.cpp Node.cpp Utility.cpp
	echo "1" | perf record -g -o perf_sequential_analysis.prof ./sequential 
	perf report -i perf_sequential_analysis.prof
#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
omp: kdtree_omp.cpp Node.cpp Node.hpp
	$(CXX) $(CXX_FLAGS) $(OPENMP) -o omp kdtree_omp.cpp Node.cpp Utility.cpp

run_omp:
	./omp

gprof_omp: kdtree_omp.cpp Node.cpp Node.hpp
	$(CXX) $(CXX_FLAGS) $(OPENMP) -pg -o omp kdtree_omp.cpp Node.cpp Utility.cpp
	echo "1" | ./omp 
	gprof omp gmon.out > gprof_omp_analysis.prof


perf_omp: kdtree_omp.cpp Node.cpp Node.hpp
	$(CXX) $(CXX_FLAGS) $(OPENMP) -o omp kdtree_omp.cpp Node.cpp Utility.cpp
	echo "1" | perf record -g -o perf_omp_analysis.prof ./omp 
	perf report -i perf_omp_analysis.prof
#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
mpi: kdtree_mpi.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) -o mpi kdtree_mpi.cpp Node.cpp Utility.cpp

run_mpi:
	mpirun -np 16 --oversubscribe ./mpi

gprof_mpi: kdtree_mpi.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) -pg -o mpi kdtree_mpi.cpp Node.cpp Utility.cpp
	echo "1" | mpirun -np 16 --oversubscribe ./mpi
	gprof mpi gmon.out > gprof_mpi_analysis.prof

perf_mpi: kdtree_mpi.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) -pg -o mpi kdtree_mpi.cpp Node.cpp Utility.cpp
	echo "1" | perf record -g -o perf_mpi_analysis.prof mpirun -np 16 --oversubscribe ./mpi
	perf report -i perf_mpi_analysis.prof
#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
hybrid: kdtree_hybrid.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o hybrid kdtree_hybrid.cpp Node.cpp Utility.cpp

run_hybrid:
	mpirun -np 4 --oversubscribe ./hybrid

gprof_hybrid: kdtree_hybrid.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -pg -o hybrid kdtree_hybrid.cpp Node.cpp Utility.cpp
	echo "1" | mpirun -np 4 --oversubscribe ./hybrid
	gprof hybrid gmon.out > gprof_hybrid_analysis.prof

perf_hybrid: kdtree_hybrid.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o hybrid kdtree_hybrid.cpp Node.cpp Utility.cpp
	echo "1" | perf record -g -o perf_hybrid_analysis.prof mpirun -np 4 --oversubscribe ./hybrid
	perf report -i perf_hybrid_analysis.prof
#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
bonus: kdtree_bonus.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o bonus kdtree_bonus.cpp Node.cpp Utility.cpp

run_bonus:
	mpirun -np 4 --oversubscribe ./bonus

gprof_bonus: kdtree_bonus.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -pg -o bonus kdtree_hybrid.cpp Node.cpp Utility.cpp
	echo "1" | mpirun -np 4 --oversubscribe ./bonus
	gprof bonus gmon.out > gprof_bonus_analysis.prof

perf_bonus: kdtree_bonus.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o bonus kdtree_bonus.cpp Node.cpp Utility.cpp
	echo "1" | perf record -g -o perf_bonus_analysis.prof mpirun -np 4 --oversubscribe ./bonus
	perf report -i perf_bonus_analysis.prof

perf_bonus_server: kdtree_bonus.cpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o bonus kdtree_bonus.cpp Node.cpp Utility.cpp
	echo "1" | perf record -g -o perf_bonus_analysis.prof mpirun -np 4 --oversubscribe ./bonus
	perf report -i perf_bonus_analysis.prof > perf_bonus_analysis.text
#-----------------------------------------------------------------------------------------#

clean:
	rm -rf *.o sequential omp mpi hybrid bonus *.out

distclean:
	rm -rf *.o sequential omp mpi hybrid bonus *.out *.prof*