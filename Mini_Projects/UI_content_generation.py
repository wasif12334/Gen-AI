import streamlit as st

from Content_genreator import (
    model,
    prompt,
    parser,
)

st.set_page_config(
    page_title="AI Content Generator",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 AI Content Generator")
st.write("Generate SEO-Optimized Content for YouTube & Social Media")

title = st.text_input("Title")

platform = st.selectbox(
    "Platform",
    [
        "YouTube",
        "Instagram",
        "TikTok",
        "Facebook",
        "LinkedIn",
        "X (Twitter)",
        "Blog",
    ],
)

category = st.selectbox(
    "Content Category",
    [
        "Education",
        "Technology",
        "Gaming",
        "Entertainment",
        "Finance",
        "Business",
        "Lifestyle",
        "Travel",
        "Health",
        "Sports",
        "News",
        "Other",
    ],
)

audience = st.text_input("Target Audience")

tone = st.selectbox(
    "Tone",
    [
        "Professional",
        "Friendly",
        "Casual",
        "Luxury",
        "Motivational",
        "Educational",
        "Funny",
    ],
)

language = st.selectbox(
    "Language",
    [
        "English",
        "Urdu",
        "Hindi",
        "Arabic",
        "Spanish",
        "French",
    ],
)

if st.button("Generate Content"):

    with st.spinner("Generating Content..."):

        final_prompt = prompt.invoke(
            {
                "title": title,
                "platform": platform,
                "category": category,
                "audience": audience,
                "tone": tone,
                "language": language,
                "format_instruction": parser.get_format_instructions(),
            }
        )

        response = model.invoke(final_prompt)

        data = parser.parse(response.content[0]["text"])

    st.success("Content Generated Successfully!")

    st.subheader("Short Description")
    st.write(data.Short_Description)

    st.subheader("Long Description")
    st.write(data.Long_Description)

    st.subheader("Relevant Keywords")
    st.write(", ".join(data.Relevant_keywords))

    st.subheader("SEO Tags")
    st.write(data.SEO_Tags)

    st.subheader("Video Category")
    st.write(", ".join(data.Video_Category))

    st.subheader("Suggested Thumbnail Text")
    st.write(data.Suggested_Thumbnail_Text)

    st.subheader("Call To Action")
    st.write(data.Call_To_Action)

    st.subheader("Alternative Titles")
    for title in data.Alternative_Titles:
        st.write(f"• {title}")

    st.subheader("Tips to Improve Discoverability")
    st.write(data.Tips_to_Improve_Discoverability)