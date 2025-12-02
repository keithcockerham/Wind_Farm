# GitHub Pages Setup Instructions

## Quick Setup (5 minutes)

### 1. Upload to Your GitHub Repository

```bash
# In your wind-turbine-prediction repo
mkdir -p docs
cp -r /path/to/these/files/* docs/

git add docs/
git commit -m "Add GitHub Pages portfolio site"
git push origin main
```

### 2. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Source":
   - Branch: `main`
   - Folder: `/docs`
4. Click **Save**

### 3. Access Your Site

Your site will be live at:
```
https://yourusername.github.io/wind-turbine-prediction/
```

(Replace `yourusername` with your GitHub username)

## Files Included

```
docs/
├── index.html           # Main portfolio page
├── methodology.html     # (Create this next if needed)
├── results.html         # (Create this next if needed)
├── css/
│   └── style.css       # All styling
└── js/
    └── charts.js       # Interactive Plotly charts
```

## Customization

### Update Your Information

In `index.html`, replace:
- `yourusername` → your GitHub username (lines 16, 114, 619)
- Add your LinkedIn URL (line 619)
- Update GitHub repo URL (lines 16, 114)

### Add More Pages

To add methodology.html or results.html:
1. Copy the structure from index.html
2. Replace the content sections
3. Keep the same navigation and footer

## Benefits

✅ **100% Reliable** - Served by GitHub's CDN
✅ **Fast Loading** - Static files, no server processing  
✅ **No Authentication Issues** - Public by default
✅ **Version Controlled** - Full Git history
✅ **Custom Domain** - Can add your own domain later
✅ **Free Forever** - GitHub Pages is free for public repos

## Testing Locally

Open `index.html` in your browser to test before pushing.

## Need Help?

If you encounter issues:
1. Check that files are in `docs/` folder
2. Verify GitHub Pages is enabled in Settings
3. Wait 2-3 minutes for first deployment
4. Check https://github.com/yourusername/repo/deployments

## Next Steps

1. ✅ Upload files
2. ✅ Enable GitHub Pages  
3. ✅ Test your site
4. Create methodology.html and results.html
5. Add your JSON results files to docs/data/
6. Share your portfolio URL!

