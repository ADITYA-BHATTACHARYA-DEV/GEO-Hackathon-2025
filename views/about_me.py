# import streamlit as st
#
# # ✅ Use the new dialog API
# @st.dialog("Contact Me")
# def show_contact_form():
#     st.text_input("First Name")
#     st.text_input("Last Name")
#     st.text_input("Email")
#     st.number_input("Phone Number")
#     st.text_input("Message")
#     bt = st.button("Submit 🚀")
#     if bt:
#         st.success("Message sent Successfully !!")
#
# st.title("Individual Profiles", anchor=False)
# # --- Layout ---
# col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
#
# with col1:
#     st.image("./assets/group.png")
#
# with col2:
#     st.title("Rajiv Gandhi Institute of Petroleum Technology, Jais, India", anchor=False)
#     st.write("For the Bridging Innovation between Petroleum and Data Science with Modern AI")
#
#     # 🖥️ Button to open dialog
#     if st.button("🖥️ Contact Us"):
#         show_contact_form()
#
#
# with col1:
#     st.image("./assets/s.png")
#
# with col2:
#     st.title("Aditya Bhattacharya", anchor=False)
#     st.write("Aspiring Software Engineer and Developer")
#
#     # 🖥️ Button to open dialog
#     if st.button("🖥️ Contact Me"):
#         show_contact_form()
#
# st.write("\n")
#
# # --- Info Section ---
# col1, col2 = st.columns(2, gap="small")
#
# with col1:
#     st.subheader("Experience & Qualification", anchor=False)
#     st.write(
#         "Undergraduate student in B.Tech Computer Science, Rajiv Gandhi Institute of Petroleum Technology"
#     )
#     st.write(
#         """
#     - Strong hands-on experience and knowledge in Machine / Deep Learning, Full Stack Web / App Development
#     - Good understanding of statistical principles and their respective applications
#     - Excellent team-player and displaying a strong sense of initiative
#     """
#     )
#
# with col2:
#     st.image(
#         "./views/Rajiv_Gandhi_Institute_of_Petroleum_Technology-removebg-preview.png",
#         width=50,
#     )
#     st.subheader("Proficient in Tech Stacks:", anchor=False)
#     st.image("./assets/img.png")
#
#
# ####----Rudra------####
# with col1:
#     st.image("./assets/rudra.jpg")
#
# with col2:
#     st.title("Rudradip Khanra", anchor=False)
#     st.write("Aspiring Software Engineer and Developer")
#
#
#
# st.write("\n")
#
# # --- Info Section ---
# col1, col2 = st.columns(2, gap="small")
#
# with col1:
#     st.subheader("Experience & Qualification", anchor=False)
#     st.write(
#         "Undergraduate student in B.Tech Computer Science, Rajiv Gandhi Institute of Petroleum Technology"
#     )
#     st.write(
#         """
#     - Strong hands-on experience and knowledge in Machine / Deep Learning, Full Stack Web / App Development
#     - Good understanding of statistical principles and their respective applications
#     - Excellent team-player and displaying a strong sense of initiative
#     """
#     )
# with col2:
#     st.image(
#         "./views/Rajiv_Gandhi_Institute_of_Petroleum_Technology-removebg-preview.png",
#         width=50,
#     )
#     st.subheader("Proficient in Tech Stacks:", anchor=False)
#     st.image("./assets/img.png")
#
# ####---Mohit----#######
# with col1:
#     st.image("./assets/Mohit.jpg")
#
# with col2:
#     st.title("Mohit Singh", anchor=False)
#     st.write("Aspiring Software Engineer and Developer")
#
#
#
# st.write("\n")
#
# # --- Info Section ---
# col1, col2 = st.columns(2, gap="small")
#
# with col1:
#     st.subheader("Experience & Qualification", anchor=False)
#     st.write(
#         "Undergraduate student in B.Tech Computer Science, Rajiv Gandhi Institute of Petroleum Technology"
#     )
#     st.write(
#         """
#     - Strong hands-on experience and knowledge in Machine / Deep Learning, Full Stack Web / App Development
#     - Good understanding of statistical principles and their respective applications
#     - Excellent team-player and displaying a strong sense of initiative
#     """
#     )
#
#
# ####-----Shreyansh-----###
# with col1:
#     st.image("./assets/shrey.jpg")
#
# with col2:
#     st.title("Shreyansh Gupta", anchor=False)
#     st.write("Aspiring Software Engineer and Developer")
#
#
#
# st.write("\n")
#
# # --- Info Section ---
# col1, col2 = st.columns(2, gap="small")
#
# with col1:
#     st.subheader("Experience & Qualification", anchor=False)
#     st.write(
#         "Undergraduate student in B.Tech Computer Science, Rajiv Gandhi Institute of Petroleum Technology"
#     )
#     st.write(
#         """
#     - Strong hands-on experience and knowledge in Machine / Deep Learning, Full Stack Web / App Development
#     - Good understanding of statistical principles and their respective applications
#     - Excellent team-player and displaying a strong sense of initiative
#     """
#     )


import streamlit as st

# ✅ Use the new dialog API
@st.dialog("Contact Me")
def show_contact_form():
    st.text_input("First Name")
    st.text_input("Last Name")
    st.text_input("Email")
    st.number_input("Phone Number")
    st.text_input("Message")
    bt = st.button("Submit 🚀")
    if bt:
        # Note: In a real app, you'd process the form data here
        st.success("Message sent Successfully !!")

st.title("Individual Profiles", anchor=False)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(
        "./assets/rg.png",
        width=1600
    )

# Optional: Using a simpler 3-column split for basic centering
# col_left, col_center, col_right = st.columns(3)

# with col_center:
#     st.image(
#         "./views/Rajiv_Gandhi_Institute_of_Petroleum_Technology-removebg-preview.png",
#         width=200
#     )
# --- Institute / Group Header ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    st.image("./assets/group.png")

with col2:
    st.title("Team:-- \n", anchor=False)
    st.title("QuadPetro-AI, India", anchor=False)
    st.write("For the Bridging Innovation between Petroleum and Data Science with Modern AI")

    # 🖥️ Button to open dialog
    if st.button("🖥️ Contact Us"):
        show_contact_form()

st.write("---") # Visual separator

# ===============================
# 👨‍💻 Aditya Bhattacharya Profile
# ===============================
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    st.image("./assets/s.png")

with col2:
    st.title("Aditya Bhattacharya", anchor=False)
    st.write("Aspiring Software Engineer and Developer")

    # 🖥️ Button to open dialog
    if st.button("🖥️ Contact Me"):
        show_contact_form()

st.write("\n")

# --- Info Section for Aditya ---
col_info_1, col_info_2 = st.columns(2, gap="small")

with col_info_1:
    st.subheader("Experience & Qualification", anchor=False)
    st.write(
        "Undergraduate student in B.Tech Computer Science, Rajiv Gandhi Institute of Petroleum Technology"
    )
    st.write(
        """
    - Strong hands-on experience and knowledge in Machine / Deep Learning, Full Stack Web / App Development
    - Good understanding of statistical principles and their respective applications
    - Excellent team-player and displaying a strong sense of initiative
    """
    )

with col_info_2:
    st.image(
        "./views/Rajiv_Gandhi_Institute_of_Petroleum_Technology-removebg-preview.png",
        width=50,
    )
    st.subheader("Proficient in Tech Stacks:", anchor=False)
    st.image("./assets/img.png")

st.write("---") # Visual separator


# ===============================
# 👨‍💻 Mohit Singh Profile
# ===============================
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    st.image("./assets/Mohit.jpg")

with col2:
    st.title("Mohit Singh", anchor=False)
    st.write("Aspiring Software Engineer and Developer")

    # 🖥️ Button to open dialog
    if st.button("🖥️ Contact Mohit"):
        show_contact_form()

st.write("\n")

# --- Info Section for Mohit ---
col_info_1, col_info_2 = st.columns(2, gap="small")

with col_info_1:
    st.subheader("Experience & Qualification", anchor=False)
    st.write(
        "Undergraduate student in B.Tech Computer Science, Rajiv Gandhi Institute of Petroleum Technology"
    )
    st.write(
        """
    - Strong hands-on experience and knowledge in Machine / Deep Learning, Full Stack Web / App Development
    - Good understanding of statistical principles and their respective applications
    - Excellent team-player and displaying a strong sense of initiative
    """
    )
with col_info_2:
    # Adding the tech stack/logo section for Mohit, which was missing in the original code
    st.image(
        "./views/Rajiv_Gandhi_Institute_of_Petroleum_Technology-removebg-preview.png",
        width=50,
    )
    st.subheader("Proficient in Tech Stacks:", anchor=False)
    st.image("./assets/img.png")


st.write("---") # Visual separator


# ===============================
# 👨‍💻 Rudradip Khanra Profile
# ===============================
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    st.image("./assets/rudra.jpg")

with col2:
    st.title("Rudradip Khanra", anchor=False)
    st.write("Aspiring Software Engineer and Developer")

    # 🖥️ Button to open dialog
    if st.button("🖥️ Contact Rudradip"):
        show_contact_form()

st.write("\n")

# --- Info Section for Rudradip ---
col_info_1, col_info_2 = st.columns(2, gap="small")

with col_info_1:
    st.subheader("Experience & Qualification", anchor=False)
    st.write(
        "Undergraduate student in B.Tech Computer Science, Rajiv Gandhi Institute of Petroleum Technology"
    )
    st.write(
        """
    - Strong hands-on experience and knowledge in Machine / Deep Learning, Full Stack Web / App Development
    - Good understanding of statistical principles and their respective applications
    - Excellent team-player and displaying a strong sense of initiative
    """
    )
with col_info_2:
    st.image(
        "./views/Rajiv_Gandhi_Institute_of_Petroleum_Technology-removebg-preview.png",
        width=50,
    )
    st.subheader("Proficient in Tech Stacks:", anchor=False)
    st.image("./assets/img.png")

st.write("---") # Visual separator
# ===============================
# 👨‍💻 Shreyansh Gupta Profile
# ===============================
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    st.image("./assets/shrey.jpg")

with col2:
    st.title("Shreyansh Gupta", anchor=False)
    st.write("Aspiring Software Engineer and Developer")

    # 🖥️ Button to open dialog
    # I've used a unique key for this button to avoid Streamlit's DuplicateWidgetID error
    if st.button("🖥️ Contact Shreyansh"):
        show_contact_form()

st.write("\n")

# --- Info Section for Shreyansh ---
col_info_1, col_info_2 = st.columns(2, gap="small")

with col_info_1:
    st.subheader("Experience & Qualification", anchor=False)
    st.write(
        "Undergraduate student in B.Tech Computer Science, Rajiv Gandhi Institute of Petroleum Technology"
    )
    st.write(
        """
    - Strong hands-on experience and knowledge in Machine / Deep Learning, Full Stack Web / App Development
    - Good understanding of statistical principles and their respective applications
    - Excellent team-player and displaying a strong sense of initiative
    """
    )
with col_info_2:
    # Adding the tech stack/logo section for Shreyansh, which was missing in the original code
    st.image(
        "./views/Rajiv_Gandhi_Institute_of_Petroleum_Technology-removebg-preview.png",
        width=50,
    )
    st.subheader("Proficient in Tech Stacks:", anchor=False)
    st.image("./assets/img.png")