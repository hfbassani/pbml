/* 
 * File:   Cluster.h
 * Author: hans
 *
 * Created on 16 de Março de 2012, 13:38
 */

#ifndef CLUSTER_H
#define	CLUSTER_H

#include <vector>
#include <algorithm>

class Cluster {

public:
        std::vector<bool> m_subspace;
        std::vector<int> m_objects;

	Cluster(std::vector<bool> subspace, std::vector<int> objects) {
		m_subspace = subspace;
		m_objects = objects;
	}

	bool equals(Cluster cluster) {
		if (m_subspace.size() != cluster.m_subspace.size())
			return false;
		for (int i = 0; i < m_subspace.size(); i++)
			if (m_subspace[i] != cluster.m_subspace[i])
				return false;
                
                std::sort(m_objects.begin(), m_objects.end());
                std::sort(cluster.m_objects.begin(), cluster.m_objects.end());
                
                std::vector<int>::iterator itm = m_objects.begin();
                std::vector<int>::iterator itC = cluster.m_objects.begin();
                for (;itm!=m_objects.end(); itm++, itC++) {
                    if ((*itm) != (*itC))
                        return false;
                }
		return true;
	}
        
        /*

	 String toString() {
		StringBuffer buf = new StringBuffer();
		for (bool value : m_subspace)
			buf.append((value) ? "1 " : "0 ");
		// buf.append(":")
		buf.append(m_objects.size() + " ");
		for (int value : m_objects)
			buf.append(value + " ");
		buf.append("\n");
		return buf.toString();
	}

	 String toString2() {
		StringBuffer buf = new StringBuffer();
		for (bool value : m_subspace)
			buf.append((value) ? "1 " : "0 ");
		buf.append("#: ").append(m_objects.size());
		buf.append("\n");
		return buf.toString();
	}

	 String export() {
		StringBuffer buf = new StringBuffer();
		for (bool value : m_subspace)
			buf.append((value) ? "true " : "false ");
		for (int value : m_objects)
			buf.append(value + " ");
		buf.append("\n");
		return buf.toString();
	}

	 String toString3() {
		StringBuffer buf = new StringBuffer();
		for (bool value : m_subspace)
			buf.append((value) ? "1 " : "0 ");
		buf.append("#: ").append(m_objects.size()+" / ");
		for(int i=0;i<m_objects.size();i++)
			buf.append(m_objects.get(i)+" ");
		buf.append("\n");
		return buf.toString();
	}
	
	 String toStringWeka() {
		StringBuffer buf = new StringBuffer();
		buf.append("[");
		for (bool value : m_subspace)
			buf.append((value) ? "1 " : "0 ");
		buf.append("] #"+m_objects.size()+" {");

		for (int value : m_objects)
			buf.append(value + " ");
		
		buf.append("}\n");
		return buf.toString();
	}
        */
};

#endif	/* CLUSTER_H */

