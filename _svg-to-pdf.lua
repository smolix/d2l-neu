-- Lua filter: replace .svg image paths with .pdf for LaTeX/PDF output,
-- but only if the .pdf version exists (pre-converted by convert_svg.sh).
-- External URLs and missing PDFs are left as-is for Pandoc to handle.
if FORMAT:match 'latex' or FORMAT:match 'pdf' then
  function Image(img)
    if img.src:match('%.svg$') and not img.src:match('^https?://') then
      local pdf_path = img.src:gsub('%.svg$', '.pdf')
      local f = io.open(pdf_path, 'r')
      if f then
        f:close()
        img.src = pdf_path
      end
    end
    return img
  end
end
