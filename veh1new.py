import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import math
import folium
from folium import plugins
from streamlit_folium import st_folium
import urllib.request
import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 🌿 นำเข้าไลบรารีสำหรับการประมวลผล GIS ขั้นสูง
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString

# =====================================================================
# 🛠️ 1. การตั้งค่าทรัพยากรและฟอนต์
# =====================================================================
@st.cache_resource
def setup_thai_font():
    font_url = "https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf"
    font_path = "thsarabunnew-webfont.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(font_url, font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'TH Sarabun New'
    plt.rcParams['axes.unicode_minus'] = False

setup_thai_font()

# =====================================================================
# ⚙️ 2. ฐานข้อมูลตั้งต้น
# =====================================================================
DEFAULT_DATA = [
    ("Depot โรงจัดการขยะ", 14.862939, 102.027903, 0.0),
    ("ภูมิทัศน์(ใหม่)", 14.86903, 102.02135, 0.3),
    ("สวนพฤกษศาสตร์", 14.86991, 102.022113, 0.3),
    ("อุทยานผีเสื้อ", 14.871074, 102.022713, 0.3),
    ("ซินโครตรอน", 14.872731, 102.023232, 0.1),
    ("อาคารสุรพัฒน์ 2", 14.8754, 102.02286, 0.2),
    ("เซเว่น-อีเลฟเว่น เทคโนธานี", 14.876072, 102.022745, 0.7),
    ("หอดูดาว", 14.87414, 102.027598, 0.2),
    ("กาญจนาภิเษก", 14.873602, 102.026147, 0.5)
]

# =====================================================================
# 🧮 3. ฟังก์ชันคณิตศาสตร์ (Haversine) และ QGIS NetworkX
# =====================================================================
def haversine_dist(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * (2 * math.asin(math.sqrt(a)))

@st.cache_data
def build_qgis_graph(geojson_path):
    G = nx.Graph()
    try:
        gdf = gpd.read_file(geojson_path)
        
        # ⚠️ แก้ปัญหาที่ 1: บังคับแปลงพิกัด (CRS) ให้เป็น Lat/Lon (EPSG:4326) เสมอ
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
            
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom and geom.geom_type in ['LineString', 'MultiLineString']:
                lines = [geom] if geom.geom_type == 'LineString' else geom.geoms
                for line in lines:
                    coords = list(line.coords)
                    for i in range(len(coords)-1):
                        p1, p2 = coords[i], coords[i+1]
                        dist = haversine_dist(p1[0], p1[1], p2[0], p2[1])
                        G.add_edge(p1, p2, weight=dist)
        return G
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านกราฟ QGIS: {e}")
        return None

def get_nearest_node(G, point):
    nearest_node = None
    min_dist = float('inf')
    if G is None or len(G.nodes) == 0:
        return point, 0
    for node in G.nodes:
        dist = haversine_dist(point[0], point[1], node[0], node[1])
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node, min_dist

def get_distance_matrix_qgis(G, locations):
    N = len(locations)
    distance_matrix = np.zeros((N, N))
    raw_coords = [(item[2], item[1]) for item in locations]
    snapped_nodes = [get_nearest_node(G, pt)[0] for pt in raw_coords]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                try:
                    if G is not None:
                        dist = nx.shortest_path_length(G, source=snapped_nodes[i], target=snapped_nodes[j], weight='weight')
                    else:
                        dist = haversine_dist(raw_coords[i][0], raw_coords[i][1], raw_coords[j][0], raw_coords[j][1])
                    distance_matrix[i][j] = dist
                except nx.NetworkXNoPath:
                    distance_matrix[i][j] = haversine_dist(raw_coords[i][0], raw_coords[i][1], raw_coords[j][0], raw_coords[j][1])
    return pd.DataFrame(distance_matrix)

# =====================================================================
# 📡 4. การเชื่อมต่อโครงข่ายถนนสาธารณะ (OSRM API)
# =====================================================================
@st.cache_data(show_spinner=False)
def get_distance_matrix_osrm(locations):
    N = len(locations)
    distance_matrix = np.zeros((N, N))
    CHUNK_SIZE = 50
    coords = [(item[2], item[1]) for item in locations]
    
    for i in range(0, N, CHUNK_SIZE):
        for j in range(0, N, CHUNK_SIZE):
            src_chunk = coords[i:i+CHUNK_SIZE]
            dst_chunk = coords[j:j+CHUNK_SIZE]
            combined_coords = src_chunk + dst_chunk
            coords_string = ";".join([f"{lon},{lat}" for lon, lat in combined_coords])
            sources_str = ";".join([str(x) for x in range(len(src_chunk))])
            destinations_str = ";".join([str(x) for x in range(len(src_chunk), len(src_chunk) + len(dst_chunk))])
            
            url = f"http://router.project-osrm.org/table/v1/driving/{coords_string}?sources={sources_str}&destinations={destinations_str}&annotations=distance"
            try:
                response = requests.get(url)
                data = response.json()
                if data.get("code") == "Ok":
                    distance_matrix[i:i+len(src_chunk), j:j+len(dst_chunk)] = np.array(data["distances"])
            except Exception:
                pass
            time.sleep(0.5)
    return pd.DataFrame(distance_matrix) / 1000.0

# =====================================================================
# 🧠 5. อัลกอริทึมการจัดเส้นทาง (Optimization Engines)
# =====================================================================
def run_savings_algorithm(df_dist, demands, nodes, max_capacity):
    depot = nodes[0]
    customers = nodes[1:]
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                s_ij = df_dist.loc[i, depot] + df_dist.loc[depot, j] - df_dist.loc[i, j]
                if s_ij > 0: savings.append((s_ij, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    routes = [[c] for c in customers]
    route_vols = [demands[c] for c in customers]

    def get_route_idx(node):
        for idx, r in enumerate(routes):
            if node in r: return idx
        return -1

    for s_ij, i, j in savings:
        idx_i, idx_j = get_route_idx(i), get_route_idx(j)
        if idx_i != idx_j and idx_i != -1 and idx_j != -1:
            if routes[idx_i][-1] == i and routes[idx_j][0] == j:
                if route_vols[idx_i] + route_vols[idx_j] <= max_capacity:
                    routes[idx_i].extend(routes[idx_j])
                    route_vols[idx_i] += route_vols[idx_j]
                    routes.pop(idx_j)
                    route_vols.pop(idx_j)
    return routes, route_vols

def run_sweep_algorithm(locations, demands, nodes, max_capacity):
    depot, depot_lat, depot_lon = nodes[0], locations[0][1], locations[0][2]
    customer_angles = []
    for i, item in enumerate(locations[1:]):
        node_name, lat, lon = item[0], item[1], item[2]
        angle = math.degrees(math.atan2(lat - depot_lat, lon - depot_lon))
        if angle < 0: angle += 360
        customer_angles.append({"node": node_name, "angle": angle, "vol": demands[node_name]})
        
    customer_angles.sort(key=lambda x: x['angle'])
    routes, route_vols, current_route, current_vol = [], [], [], 0.0
    
    for c in customer_angles:
        if current_vol + c['vol'] <= max_capacity:
            current_route.append(c['node'])
            current_vol += c['vol']
        else:
            routes.append(current_route)
            route_vols.append(current_vol)
            current_route = [c['node']]
            current_vol = c['vol']
            
    if current_route:
        routes.append(current_route)
        route_vols.append(current_vol)
    return routes, route_vols

# =====================================================================
# 🗺️ 6. โมดูลสร้างแผนที่ Interactive (Folium)
# =====================================================================
def create_interactive_map(routes, locations, nodes, routing_mode, G_qgis=None, qgis_file_path=None):
    depot_coords = (locations[0][1], locations[0][2])
    m = folium.Map(location=depot_coords, zoom_start=16, tiles='CartoDB positron')
    
    # วาดเส้นถนน QGIS
    if routing_mode == "QGIS Custom Map (ออฟไลน์)" and qgis_file_path is not None and os.path.exists(qgis_file_path):
        try:
            gdf_bg = gpd.read_file(qgis_file_path)
            if gdf_bg.crs is None: gdf_bg.set_crs(epsg=4326, inplace=True)
            elif gdf_bg.crs.to_epsg() != 4326: gdf_bg = gdf_bg.to_crs(epsg=4326)
            
            # --- แก้ไข: เพิ่มการหา Bounds และซูมไปที่ถนน ---
            bounds = gdf_bg.total_bounds # [minx, miny, maxx, maxy]
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            
            folium.GeoJson(
                gdf_bg.to_json(),
                name="โครงข่ายถนน QGIS",
                style_function=lambda x: {'color': '#000000', 'weight': 3, 'opacity': 0.7}
            ).add_to(m)
            st.sidebar.success("✅ วาดถนน QGIS สำเร็จ")
        except Exception as e:
            st.sidebar.error(f"❌ อ่านไฟล์ QGIS ไม่ได้: {e}")

    # ปักหมุด Depot
    folium.Marker(location=depot_coords, popup="DEPOT", icon=folium.Icon(color="red", icon="home")).add_to(m)
    
    # ปักหมุดจุดเก็บขยะ
    for item in locations[1:]:
        folium.CircleMarker(location=(item[1], item[2]), radius=5, color="blue", fill=True).add_to(m)

    # วาดเส้นทางรถวิ่ง
    colors = ['#FF5733', '#335BFF', '#28B463', '#9B59B6']
    coords_lonlat = {item[0]: (item[2], item[1]) for item in locations}

    for trip_idx, route_seq in enumerate(routes):
        full_route = [nodes[0]] + route_seq + [nodes[0]]
        # วาดเส้นด้วย AntPath
        path_coords = []
        for n in full_route:
            pt = coords_lonlat[n]
            path_coords.append((pt[1], pt[0])) # Lat, Lon
            
        plugins.AntPath(locations=path_coords, color=colors[trip_idx % len(colors)], weight=5).add_to(m)
        
    return m

# =====================================================================
# 🖥️ 7. หน้าจอผู้ใช้งาน (Streamlit UI)
# =====================================================================
st.set_page_config(page_title="Smart Waste Collection CVRP", layout="wide")
st.title("🚛 Smart Waste Collection Routing System (QGIS & OSRM Edition)")
st.markdown("รองรับการคำนวณเส้นทางจาก **โครงข่ายถนนส่วนตัว (QGIS GeoJSON)** และแผนที่สาธารณะ (OSRM)")

with st.sidebar:
    st.header("⚙️ 1. เลือกแหล่งข้อมูลโครงข่ายถนน")
    routing_mode = st.radio(
        "Routing Engine:",
        ("OSRM API (ออนไลน์/สาธารณะ)", "QGIS Custom Map (ออฟไลน์)")
    )
    
    qgis_file_path = "sut_roads.geojson"
    G_qgis = None
    if routing_mode == "QGIS Custom Map (ออฟไลน์)":
        if os.path.exists(qgis_file_path):
            st.success(f"✅ พบไฟล์ `{qgis_file_path}` ในระบบ")
            G_qgis = build_qgis_graph(qgis_file_path)
            if G_qgis is None or len(G_qgis.nodes) == 0:
                st.error("❌ พบไฟล์ GeoJSON แต่ไม่สามารถสกัดข้อมูลเส้นถนนได้ (อาจวาดผิดฟอร์แมต หรือไม่มี LineString)")
        else:
            st.warning(f"⚠️ ไม่พบไฟล์ `{qgis_file_path}`! กรุณานำไฟล์ที่ Export จาก QGIS มาวางไว้ในโฟลเดอร์เดียวกับโปรแกรม")

    st.header("⚙️ 2. ปรับแต่งยานพาหนะ (Fleet)")
    max_vehicles = st.number_input("จำนวนรถขยะที่มีในระบบ", min_value=1, value=2)
    max_capacity = st.number_input("ความจุสูงสุดของรถ (ลบ.ม.)", min_value=1.0, value=4.5)
    
    st.header("⚙️ 3. เลือกอัลกอริทึม")
    algorithm_choice = st.selectbox("เทคนิคการจัดเส้นทาง", ("Clarke-Wright Savings", "Sweep Algorithm"))
    
    st.header("🌿 4. ตัวแปรเศรษฐศาสตร์และคาร์บอน")
    fuel_economy = st.number_input("อัตราสิ้นเปลือง (กม./ลิตร)", value=5.0)
    fuel_price = st.number_input("ราคาน้ำมัน (บาท/ลิตร)", value=32.94)
    ef_value = st.number_input("ค่า EF (kgCO₂/ลิตร)", value=2.7446, format="%.4f")
    gwp_value = st.number_input("ค่า GWP", value=1.0)

# --- ส่วนจัดการตารางข้อมูล ---
st.subheader("📝 ตารางจัดการข้อมูลพิกัด GPS และปริมาณขยะ")
df_input = pd.DataFrame(DEFAULT_DATA, columns=["Node_Name", "Latitude", "Longitude", "Demand"])
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
start_btn = st.button("🚀 ยืนยันข้อมูลและเริ่มการประมวลผล", type="primary")

# =====================================================================
# 🚀 8. ส่วนประมวลผลหลัก (Execution Block)
# =====================================================================
if start_btn:
    if routing_mode == "QGIS Custom Map (ออฟไลน์)" and not os.path.exists(qgis_file_path):
        st.error(f"❌ ไม่สามารถประมวลผลได้ เนื่องจากไม่พบไฟล์ `{qgis_file_path}`")
        st.stop()

    data_to_use = edited_df.values.tolist()
    nodes, demands = edited_df["Node_Name"].tolist(), dict(zip(edited_df["Node_Name"], edited_df["Demand"]))
    osrm_input_format = [(row[0], row[1], row[2], row[3]) for row in data_to_use]

    with st.spinner(f"📡 กำลังคำนวณระยะทางจาก {routing_mode}..."):
        if routing_mode == "QGIS Custom Map (ออฟไลน์)":
            df_dist = get_distance_matrix_qgis(G_qgis, osrm_input_format)
        else:
            df_dist = get_distance_matrix_osrm(osrm_input_format)
            
        df_dist.columns = df_dist.index = nodes

        if algorithm_choice == "Clarke-Wright Savings":
            routes, route_vols = run_savings_algorithm(df_dist, demands, nodes, max_capacity)
        else:
            routes, route_vols = run_sweep_algorithm(osrm_input_format, demands, nodes, max_capacity)

        grand_total_distance = sum(sum(df_dist.loc[full_route[k], full_route[k+1]] for k in range(len(full_route)-1)) for full_route in ([nodes[0]] + r + [nodes[0]] for r in routes))
        grand_total_volume = sum(route_vols)
        
        activity_data_A = grand_total_distance / fuel_economy
        carbon_emitted_E = activity_data_A * ef_value * gwp_value
        total_fuel_cost = activity_data_A * fuel_price
        
        st.success(f"✅ ประมวลผลผ่าน {routing_mode} สำเร็จ!")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("รอบวิ่ง (Trips)", f"{len(routes)} เที่ยว")
        col2.metric("ระยะทางจริง", f"{grand_total_distance:.2f} กม.")
        col3.metric("ปริมาตรขยะ", f"{grand_total_volume:.2f} ลบ.ม.")
        col4.metric("คาร์บอน (CO₂e)", f"{carbon_emitted_E:.2f} kg")
        col5.metric("ต้นทุนน้ำมัน", f"฿ {total_fuel_cost:,.2f}")
        
        with st.spinner("🗺️ กำลังสร้างแผนที่ GPS Interactive (พร้อมแสดงโครงข่าย QGIS)..."):
            m = create_interactive_map(routes, osrm_input_format, nodes, routing_mode, G_qgis, qgis_file_path)
            st_folium(m, width=1200, height=600)
            
        st.markdown("### 📋 ตารางการปฏิบัติงานแยกตามยานพาหนะ")
        fleet_schedule = {f"🚛 รถขยะคันที่ {i+1}": [] for i in range(int(max_vehicles))}
        for i, r in enumerate(routes):
            fleet_schedule[f"🚛 รถขยะคันที่ {(i % int(max_vehicles)) + 1}"].append({"trip_sequence": (i // int(max_vehicles)) + 1, "route": r, "vol": route_vols[i]})
        
        for vehicle_name, trips in fleet_schedule.items():
            with st.expander(f"{vehicle_name} (รับผิดชอบ {len(trips)} เที่ยววิ่ง)", expanded=True):
                for t in trips:
                    st.info(f"📍 Depot ➡️ {' ➡️ '.join(t['route'])} ➡️ Depot (ขยะ: {t['vol']:.2f} ลบ.ม.)")
