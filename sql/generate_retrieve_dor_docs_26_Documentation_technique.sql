/*
 RETRIEVE DOCUMENTS FROM FAMILY 'Documentation technique' [26] INDEXED BY 'ATELIER NUM GOELAND'
*/
SELECT --E.IdType,
       --(Select ET.Type FROM DicoTypeEnveloppe ET WHERE ET.IdType=E.IdType) as TypeEnv,
       COUNT(*) as num, IdDocFamily,
       (Select F.Name FROM DocFamily F WHERE F.IdDocFamily=D.IdDocFamily) as family,
       D.LocFileExt,
       MIN(D.IdUserPost) as min_IdUserPost ,MAX(D.IdUserPost) as max_IdUserPost,
       MIN(D.IdDocument) as min_id,MAX(D.IdDocument) as max_id,
       MIN(D.DateCreated) as min_date,MAX(D.DateCreated) as max_date
FROM Enveloppe E
INNER JOIN Employe_OrgUnit O ON O.IdEmploye=E.IdCreator
AND O.IdOrgUnit = 62
INNER JOIN EnveloppeDocument ED on E.IdEnveloppe = ED.IdEnveloppe
INNER JOIN Document D on D.IdDocument = ED.IdDocument
AND D.DateCreated > '2020-01-01'
AND D.IdDocFamily in (26)
GROUP BY --E.IdType,
         IdDocFamily,D.LocFileExt
ORDER BY 1 DESC;
/* as of 2021-02-04
   4697     9   Plan de projet  pdf
   2454     11  Plan            pdf
    1911    25  Plan de situation, élévation  pdf
       2    11  Plan    jpg
*/

-- retrieve 1200 pdf documents from family (9,11,25) 'Plna & co'


SELECT TOP 1200
        --output some info about current doc
        'echo '' doing doc id: ' + CONVERT(varchar, D.IdDocument) + ''';'+
        -- copy document on local current dir
        'scp kvm-golux:' + (SELECT REPLACE(S.Url, 'http://goeland.lausanne.ch/', '/data/goelanddocs/')
        FROM DocStorage S WHERE S.IdDocStorage= D.IdDocStorage) + D.LocFileName + '.' + D.LocFileExt
        + ' . ;'+
        -- convert the first page of the pdf to a jpeg file
        'pdftoppm -singlefile -jpeg ' + D.LocFileName + '.' + D.LocFileExt
            + ' ' + CONVERT(varchar, D.IdDocument)
        + '; mv ' + CONVERT(varchar, D.IdDocument) + '.jpg ' + CONVERT(varchar, D.IdDocument) + '.jpeg '
        + '; convert ' + CONVERT(varchar, D.IdDocument) + '.jpeg  -resize 320 resized_' + CONVERT(varchar, D.IdDocument) + '.jpeg '
FROM Enveloppe E
INNER JOIN Employe_OrgUnit O ON O.IdEmploye=E.IdCreator
AND O.IdOrgUnit = 62
INNER JOIN EnveloppeDocument ED on E.IdEnveloppe = ED.IdEnveloppe
INNER JOIN Document D on D.IdDocument = ED.IdDocument
--INNER JOIN DocStorage S on D.IdDocStorage = S.IdDocStorage
AND D.DateCreated > '2019-01-01'
AND D.IdDocFamily in (26)
AND D.LocFileExt = 'pdf'
AND E.IdType NOT IN (2,5) -- do not take env type Fax ou tél
ORDER BY 1 DESC;