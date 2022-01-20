import sqlite3
import numpy as np
import time


def get_trip_history(src_node, dst_node, edge_type,
                     timestamp, limit, conn: sqlite3.Connection):
    
    sql = '''
    select 
      eid,
      case when src_node=={0} then 0 else 1 end as reverse
    from 
      edges 
    where 
      (timestamp<{3} and src_node={0} and dst_node={1} and edge_type={2}) or
      (timestamp<{3} and src_node={1} and dst_node={0} and edge_type={2})
    order by timestamp desc
    limit {4}
    '''.format(src_node, dst_node, edge_type, timestamp, limit)

    cur = conn.execute(sql)

    return np.array(cur.fetchall())
    pass


def get_pair_history(src_node, dst_node, timestamp, limit,
                     conn: sqlite3.Connection):
    
    sql = '''
    select 
      eid,
      case when src_node=={0} then 0 else 1 end as reverse
    from 
      edges 
    where 
      (src_node={0} and dst_node={1} and timestamp<{2}) or
      (src_node={1} and dst_node={0} and timestamp<{2})
    order by timestamp desc
    limit {3}
    '''.format(src_node, dst_node, timestamp, limit)

    cur = conn.execute(sql)

    return np.array(cur.fetchall())
    pass


def get_node_history(node, timestamp, limit,
                     conn: sqlite3.Connection):
    
    sql = '''
    select 
      eid,
      case when src_node=={0} then 0 else 1 end as reverse
    from 
      edges 
    where 
      (src_node={0} and timestamp<{1}) or
      (dst_node={0} and timestamp<{1})
    order by timestamp desc, eid desc
    limit {2}
    '''.format(node, timestamp, limit)

    cur = conn.execute(sql)

    return np.array(cur.fetchall())
    pass


if __name__ == '__main__':
    conn = sqlite3.connect('../dataset_A/data.sqlite')
    start = time.time()
    eids = get_trip_history(
        507, 17549, 151,
        1497473624,
        64,
        conn)
    print(eids)
    print(time.time() - start)

    start = time.time()
    eids = get_pair_history(
        507, 17549,
        1497473624,
        64,
        conn)
    print(eids)
    print(time.time() - start)

    start = time.time()
    eids = get_node_history(
        507,
        1497473624,
        64,
        conn)
    print(eids)
    print(len(eids))
    print(time.time() - start)

    conn.close
    pass
