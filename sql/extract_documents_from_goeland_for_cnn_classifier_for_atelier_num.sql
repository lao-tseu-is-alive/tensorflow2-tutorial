SELECT COUNT(*) as num, IdDocFamily,
       (Select F.Name FROM DocFamily F WHERE F.IdDocFamily=D.IdDocFamily) as family,
       MIN(D.IdUserPost) as min_IdUserPost ,MAX(D.IdUserPost) as max_IdUserPost,
       MIN(D.IdDocument) as min_id,MAX(D.IdDocument) as max_id,
       MIN(D.DateCreated) as min_date,MAX(D.DateCreated) as max_date
FROM Document D
WHERE DateCreated > '2020-01-01' --AND IdDocFamily IN (11,25)
--AND D.IdDocument > 1591432
GROUP BY IdDocFamily
ORDER BY 1 DESC;
-- to many family are not used by 'atelier de numerisation' but only a few people
-- so let's limit to documents indexed by  'atelier de numerisation'
----------------------------------------------------------------------------------

SELECT
COUNT(*) as num, IdDocFamily,
       (Select F.Name FROM DocFamily F WHERE F.IdDocFamily=D.IdDocFamily) as family,
       COUNT(DISTINCT D.LocFileExt)as num_ext,
       MIN(D.LocFileExt) as min_LocFileExt ,MAX(D.LocFileExt) as max_LocFileExt,
       COUNT(DISTINCT D.IdUserPost) as num_IdUserPost,
       MIN(D.IdUserPost) as min_IdUserPost ,MAX(D.IdUserPost) as max_IdUserPost,
       MIN(D.IdDocument) as min_id,MAX(D.IdDocument) as max_id,
       MIN(D.DateCreated) as min_date,MAX(D.DateCreated) as max_date
FROM Enveloppe E
INNER JOIN Employe_OrgUnit O ON O.IdEmploye=E.IdCreator
AND O.IdOrgUnit = 62
INNER JOIN EnveloppeDocument ED on E.IdEnveloppe = ED.IdEnveloppe
INNER JOIN Document D on D.IdDocument = ED.IdDocument
AND D.DateCreated > '2019-01-01'
GROUP BY IdDocFamily
ORDER BY 1 DESC;


SELECT * FROM document WHERE IdDocument=1636669
--------------------------------------------------------------------------------------------------
-- retrieve 450 pdf documents from family 11 plan (run the output on a temp dir on golux)
SELECT TOP 450
'echo ' + CONVERT(varchar, D.IdDocument) +
';pdftoppm -singlefile -jpeg ' +
(SELECT REPLACE(S.Url, 'http://goeland.lausanne.ch/', '/data/goelanddocs/')
FROM DocStorage S WHERE S.IdDocStorage= D.IdDocStorage) + D.LocFileName + '.' + D.LocFileExt
+ ' ' + CONVERT(varchar, D.IdDocument)
FROM Document D
WHERE
  DateCreated > '2020-01-01' AND IdDocFamily IN (25)
  AND D.IdDocument > 1591432
  AND D.LocFileExt = 'jpg'
--D.IdDocument = 1001193
ORDER BY D.IdDocument

-- retrieve any jpg documents from family 25 plan
SELECT TOP 450
'scp kvm-golux:' +
(SELECT REPLACE(S.Url, 'http://goeland.lausanne.ch/', '/data/goelanddocs/')
FROM DocStorage S WHERE S.IdDocStorage= D.IdDocStorage) + D.LocFileName + '.' + D.LocFileExt
+ ' .'
FROM Document D
WHERE
  DateCreated > '2020-01-01' AND IdDocFamily IN (25)
  AND D.IdDocument > 1591432
  AND D.LocFileExt = 'jpg'
--D.IdDocument = 1001193
ORDER BY D.IdDocument


------------------------------------------------------------------------------
-- retrieve n pdf documents from family 25 plan_situation (run the output on a temp dir on golux)
SELECT TOP 370
'echo ' + CONVERT(varchar, D.IdDocument) +
';pdftoppm -singlefile -jpeg ' +
(SELECT REPLACE(S.Url, 'http://goeland.lausanne.ch/', '/data/goelanddocs/')
FROM DocStorage S WHERE S.IdDocStorage= D.IdDocStorage) + D.LocFileName + '.' + D.LocFileExt
+ ' ' + CONVERT(varchar, D.IdDocument)
FROM Document D
WHERE
  DateCreated > '2020-01-01' AND IdDocFamily IN (11)
  AND D.IdDocument > 1480551
  AND D.LocFileExt = 'pdf'
ORDER BY D.IdDocument

-- retrieve any jpg documents from family 25 plan_situation attention dans les jpg il y a de la merdasse
SELECT TOP 450
'scp kvm-golux:' +
(SELECT REPLACE(S.Url, 'http://goeland.lausanne.ch/', '/data/goelanddocs/')
FROM DocStorage S WHERE S.IdDocStorage= D.IdDocStorage) + D.LocFileName + '.' + D.LocFileExt
+ ' .'
FROM Document D
WHERE
  DateCreated > '2020-01-01' AND IdDocFamily IN (11)
  AND D.IdDocument > 1480551
  AND D.LocFileExt = 'jpg'
ORDER BY D.IdDocument

---------------------------------------------------------------------
-- retrieve documents from family 19 factures
SELECT TOP 45
'echo ' + CONVERT(varchar, D.IdDocument) +
';pdftoppm -singlefile -jpeg ' +
(SELECT REPLACE(S.Url, 'http://goeland.lausanne.ch/', '/data/goelanddocs/')
FROM DocStorage S WHERE S.IdDocStorage= D.IdDocStorage) + D.LocFileName + '.' + D.LocFileExt
+ ' ' + CONVERT(varchar, D.IdDocument)
FROM Document D
WHERE
  DateCreated > '2020-01-01' AND IdDocFamily IN (19)
  AND D.IdDocument > 1604167
  AND D.LocFileExt = 'pdf'
ORDER BY D.IdDocument

---------------------------------------------------------------------
-- retrieve documents from family 7 correspondance
SELECT TOP 135
'echo ' + CONVERT(varchar, D.IdDocument) +
';pdftoppm -singlefile -jpeg ' +
(SELECT REPLACE(S.Url, 'http://goeland.lausanne.ch/', '/data/goelanddocs/')
FROM DocStorage S WHERE S.IdDocStorage= D.IdDocStorage) + D.LocFileName + '.' + D.LocFileExt
+ ' ' + CONVERT(varchar, D.IdDocument)
FROM Document D
WHERE
  DateCreated > '2020-01-01' AND IdDocFamily IN (7)
  AND D.IdDocument > 1595021
  AND D.LocFileExt = 'pdf'
ORDER BY D.IdDocument

